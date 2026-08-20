# Chapter 2 Research Pass — The Capital Stack

Status: active; not cleared for writing.

## Objective

Reconstruct who is funding the current AI infrastructure expansion, through which
instruments, on whose balance sheet, and with what degree of contractual backing.

## Pass 1A — Public balance-sheet anchors

The first common-period sweep uses the latest available 2026 quarterly filings for
Microsoft, Meta, Alphabet, Amazon, and Oracle. These are not yet a comparable capex
panel: fiscal periods, lease treatment, acquisition spending, and disclosed scopes
differ. The purpose of this pass is to establish the reconciliation work required.

### Preliminary extracted observations

| Entity | Observation | Status | Amount | Why it is not directly comparable |
|---|---|---:|---:|---|
| Meta | FY2026 capex guidance including finance-lease principal | announced guidance | $125–145B | Forecast; AI efforts plus core business |
| Meta | Non-cancelable contractual commitments at 2026-03-31 | contracted aggregate | $237.67B | Includes several infrastructure categories; overlap unresolved |
| Meta | Lease obligations not yet commenced | contracted | $182.88B | Potential overlap with aggregate commitments |
| Meta | Contingent cloud-capacity purchase | upper bound | up to $14.72B | Reducible if provider resells capacity |
| Meta | April infrastructure-contract increase | subsequent event | ~$24B | Post-quarter event and likely part of later commitment balance |
| Meta | Cash restricted under purchase agreement | recognized restriction | $5.00B | Contract support, not disclosed total contract value |
| Alphabet | Q1 2026 capital expenditures | spent | $35.7B | Technical infrastructure includes AI and non-AI use |
| Alphabet | Intersect acquisition | funded acquisition | $5.862B | Includes $5.111B property, $1.214B assumed debt, and $2.170B goodwill in a non-additive allocation |
| Oracle | First nine months FY2026 capital expenditures | spent | $39.2B | Different period length; primarily data-center expansion |
| Amazon | Gross assets acquired under finance leases | recognized balance | $56.4B | Cumulative balance across several asset classes |
| Amazon/OpenAI affiliate | AWS commercial and collaboration arrangements | contracted, amount unknown | unknown | Minimum spend, term, exclusivity, and conditions undisclosed |
| CoreWeave | Q1 cash property and equipment purchases | spent | $7.695B | Cash channel only |
| CoreWeave | Q1 equipment-addition liabilities | recognized noncash | $4.592B | Includes OEM financing and later debt reclassification |
| CoreWeave | Q1 operating-lease ROU additions | recognized noncash | $1.988B | Lease channel rather than cash capex |
| CoreWeave | Aggregate debt principal | recognized/funded | $25.149B | Includes asset-level, corporate, convertible, and vendor debt |
| CoreWeave | Recognized operating lease liability | recognized | $10.050B | Present value of commenced operating leases |
| CoreWeave | Executed leases not yet commenced | contracted estimate | $40.7B | Overlap with separately disclosed site range unresolved |
| CoreWeave | Remaining performance obligations | contracted accounting backlog | $98.8B | Net of variable consideration; not cash or collateral value |

## Pass 1B — CoreWeave financing junction

CoreWeave exposes a financing chain that a hyperscaler-only capex comparison misses.
In first-quarter 2026 it combined $7.695 billion of cash equipment purchases, $4.592
billion of equipment-addition liabilities, and $1.988 billion of operating-lease
right-of-use additions. These are not three estimates of the same headline. They are
separate cash, debt/accrual, and lease channels that financed one expansion cycle.

The stock measures are equally distinct:

- $25.149 billion of debt principal, including $11.8 billion drawn under delayed-draw
  facilities and $5.036 billion of OEM/software financing;
- $10.050 billion of recognized operating lease liabilities for commenced leases;
- $40.7 billion of estimated undiscounted payments under executed leases that had not
  yet commenced;
- $98.8 billion of remaining performance obligations, net of estimated variable
  consideration, supporting but not identical to future cash receipts or debt
  collateral.

The DDTL 4.0 structure makes the project-finance mechanics visible. Its $8.5 billion
commitment was only $1.26 billion drawn at quarter end. Borrowing is sized against
eligible computing equipment, projected debt service coverage, and project conditions;
the debt is secured at a dedicated subsidiary and is non-recourse except for customary
carve-outs. This is closer to infrastructure project finance than ordinary unsecured
corporate borrowing.

### Gross reciprocal finance hidden by net presentation

CoreWeave's unnamed 78 MW data-center-service-provider arrangement is more complex than
a conventional lease:

1. the provider agreed to design, buy, build, and manage a phased facility;
2. CoreWeave bought $116 million of existing critical-infrastructure assets;
3. CoreWeave then committed up to $305 million of seven-year, 13% senior secured
   financing to the provider;
4. the provider used borrowing in part to buy those existing assets from CoreWeave;
5. CoreWeave retained control of the assets and recorded a fourteen-year financing
   obligation for the consideration received; and
6. CoreWeave separately took a fourteen-year finance lease from the same provider.

At March 31, 2026, the gross note receivable was $303 million. The financing obligation
and finance-lease liability totaled $232 million, and CoreWeave presented the receivable
net of those obligations. Net presentation is valid accounting evidence, but it is not
an adequate dependency map: the gross asset, construction, tenant, collateral, and
counterparty legs remain economically live.

This arrangement also reverses the simplistic direction of vendor finance. The future
tenant is both customer and lender to the developer. The provider uses tenant financing
to build capacity and to acquire tenant-controlled infrastructure; the tenant then owes
long-duration payments back to the provider.

### Minority ownership does not eliminate tenancy dependence

CoreWeave also held 20% of an unconsolidated VIE developing a multi-phase campus in
Kenilworth, New Jersey, after contributing $57 million of net assets. It entered a
forward-starting lease and development-management agreement with that related joint
venture. The third-party developer initially owned 85%, but remains unnamed.

This creates three CoreWeave roles—minority investor, future tenant, and development
counterparty—without consolidation or sole control. The disclosed $57 million is not
the campus cost. It is only CoreWeave's initial net-asset contribution. Total project
capital, future rent, lender claims, developer identity, and exit rights remain open.

### Multi-role NVIDIA dependency

NVIDIA occupies at least three economically distinct positions around CoreWeave:

1. supplier platform — all deployed CoreWeave GPUs at December 31 2025 were NVIDIA
   GPUs because of then-current customer-contract requirements;
2. equity capital — NVIDIA funded a $2.0 billion private placement in January 2026;
3. conditional capacity buyer — a $6.3 billion initial-value order requires NVIDIA to
   purchase qualifying residual unsold capacity through April 2032, subject to delivery,
   availability, and termination conditions.

These roles strengthen CoreWeave's funding and utilization system, but they are not
independent diversification. A shock to NVIDIA technology, supply, capital allocation,
or strategic demand can propagate through several ledgers at once.

## Pass 1C — NVIDIA as supplier, buyer, investor, and credit supporter

NVIDIA's April 26 2026 filing changes the capital map from a linear supply chain into a
partially circular system. It disclosed:

| Exposure | January 25 2026 | April 26 2026 | Status |
|---|---:|---:|---|
| Manufacturing, supply, and capacity commitments | $95.2B | $119.0B | Some agreements adjustable before firm orders |
| Multi-year cloud-service commitments | $27.0B | $30.0B | Reducible or terminable; primarily R&D use |
| Investment commitments | $11.4B | $27.0B | Contingent; expected through remainder of FY2027 |
| Partner facility-lease guarantees | not separately rolled forward here | up to $3.5B | Credit derivatives; $712M partner escrow |

During the same quarter, NVIDIA funded $18.6 billion of investments in private
companies and infrastructure funds. The company explicitly states that some of those
investees are AI model makers that may indirectly purchase or use NVIDIA products in
the cloud.

This does not prove artificial demand. It demonstrates a more precise mechanism:
NVIDIA can be exposed to the same ecosystem through manufacturing commitments,
cloud-service purchases, investee values, facility guarantees, accounts receivable, and
product revenue. Those are separate accounting categories, but counterparty or demand
shocks can correlate them.

The quarter-over-quarter increases are substantial: $23.8 billion for manufacturing
and capacity, $3.0 billion for cloud, and $15.6 billion for contingent investments.
These are changes in commitment balances, not necessarily cash spent during the
quarter.

### Accounting distinction

The $3.5 billion lease-guarantee maximum is especially instructive. NVIDIA says the
credit-derivative fair values were not material, while also disclosing a much larger
gross default exposure mitigated by $712 million of partner escrow. A small recognized
fair value therefore cannot be read as a small risk-transmission channel.

## Pass 1D — AMD customer equity incentives and partner credit support

AMD's March 2026 filing reveals a second architecture for financing demand. OpenAI OpCo
and Meta each hold a warrant to purchase as many as 160 million AMD shares for $0.01
per share. Neither warrant had vested at quarter end.

For each customer:

- the initial one-gigawatt GPU purchase is binding;
- the first warrant tranche depends on that initial deployment or shipment;
- full vesting depends on purchases scaling to six gigawatts;
- vesting also requires AMD share-price thresholds, reaching $600 for the final tranche,
  and additional technical or commercial conditions;
- purchases can flow through affiliates or qualifying authorized third parties.

The warrants therefore create explicit contingent cross-ownership: future customers
can acquire supplier equity upside if they purchase enough supplier product and the
supplier's market value performs. This is evidence of incentive alignment, not evidence
that purchases are uneconomic. It does mean that headline GPU demand and the
consideration granted to produce that demand must be analyzed together.

AMD also reported $25.662 billion of unconditional commitments across wafers,
substrates, components, cloud services, software, and technology licenses. Separately,
it disclosed $4.4 billion of uncommenced leases and $4.1 billion of maximum exposure
under commercial-partner data-center lease guarantees. The latter exceeded AMD's
$3.25 billion of funded debt principal, although a maximum contingent guarantee and
funded debt are not equivalent risk measures. The guarantee fair-value liability was
not material, again illustrating the gap between recognized accounting value and a
potential transmission pathway.

## Pass 1E — Broadcom's rack-finance transformation

Broadcom's second fiscal quarter produced the largest single-quarter commitment
discontinuity identified in this pass:

| Measure | February 1 2026 | May 3 2026 | Change |
|---|---:|---:|---:|
| Legally binding purchase commitments | $54M | $128.110B | +$128.056B |
| Firm multi-year customer RPO | not yet extracted on comparable basis | $164.6B | n/a |

The purchase commitments are enforceable obligations with fixed or minimum quantities,
price provisions, and approximate timing; cancelable arrangements and obligations with
one year or less remaining are excluded. The RPO includes a long-term custom AI
accelerator contract entered in the quarter. The filing does not identify the suppliers
or custom-AI customer behind these aggregates, so the two values cannot yet be netted or
assigned to the same contract.

The capital chain became more explicit after quarter end. On June 8, Broadcom arranged
for an investor partner to purchase AI racks based on Broadcom-designed custom
accelerators and hold related five-year customer leases. Broadcom backstopped the
customer lease obligations up to $29 billion. Exposure grows as racks deploy, declines
with customer payments, and can be mitigated through Broadcom assuming a lease or
selling the racks after default.

This is a transfer of asset ownership to an investor, but not a clean transfer of
customer credit risk. Broadcom retains a contingent obligation equal to roughly 45% of
its May 3 balance-sheet debt. That ratio is contextual only: funded corporate debt and
a maximum contingent backstop are different measures.

### Named compute pathway and unnamed finance pathway

Separately, Broadcom and Google signed agreements through up to 2031 for future custom
TPUs and next-generation AI-rack components. Their expanded collaboration with
Anthropic calls for Anthropic to access approximately 3.5 GW of next-generation
TPU-based compute beginning in 2027 through Broadcom.

The evidence does not establish that this named 3.5 GW pathway is the unnamed customer
behind the $29 billion backstop or the entire $164.6 billion RPO. The database therefore
keeps them separate and raises that match as an unresolved question rather than a
conclusion.

Broadcom's upstream concentration remains substantial: TSMC produced approximately 95%
of wafers manufactured by Broadcom's contract manufacturers during the first half of
fiscal 2026. Thus the reconstructed chain currently runs from a concentrated foundry,
through legally binding inventory commitments, into custom rack production, an
investor-owned leasing vehicle, a concentrated customer base, and a vendor backstop.

## Pass 1F — Supplier-financier failure pathways

The three semiconductor cases now resolve into different risk-transmission
architectures rather than one generic “chip shortage” narrative:

| Architecture | Upstream exposure | Demand support | Outside capital | Risk retained by supplier |
|---|---|---|---|---|
| NVIDIA ecosystem | $119.0B manufacturing, supply, and capacity commitments | cloud purchases, investee funding, and a $6.3B residual-capacity order at CoreWeave | private-company and infrastructure-fund investors | up to $3.5B partner lease guarantees plus investment and cloud commitments |
| AMD customer incentive | $25.662B unconditional commitments | two purchase-linked warrants, each up to 160M shares | commercial partners and authorized purchasing intermediaries | up to $4.1B partner lease guarantees and contingent warrant accounting |
| Broadcom rack finance | $128.110B legally binding purchase commitments and 95% foundry concentration at TSMC | $164.6B RPO and five-year rack leases | an investor partner owns deployed racks and leases | up to $29B customer lease backstop and rack disposition or lease-assumption remedies |

These rows are not comparable liability totals. They mix commitments, backlog,
contingent guarantees, and investment balances. Their analytical value is topological:
each supplier has a mechanism through which a demand shock can reach more than product
revenue.

### Reconstructed transmission sequences

**NVIDIA**

AI deployment slows  
→ GPU and networking demand weakens  
→ residual cloud capacity and investee economics deteriorate  
→ partner-facility lease performance can weaken  
→ upstream manufacturing commitments, cloud commitments, investment values, and
guarantees can become correlated exposures.

**AMD**

OpenAI or Meta deployment misses purchase milestones  
→ product volume and warrant vesting decline  
→ upstream commitments may remain  
→ partner facilities can lose utilization while AMD's lease guarantees survive.

If deployments succeed, the reverse path is also economically important: customers can
earn supplier equity upside while generating supplier product revenue. That is a
contractual incentive loop, not proof of circular or artificial demand.

**Broadcom**

Custom-AI customer payment or utilization fails  
→ investor-owned rack lease cash flow weakens  
→ Broadcom may assume leases or sell racks  
→ firm backlog conversion slows while fixed purchase commitments can remain  
→ concentrated TSMC supply and customer credit risk meet inside Broadcom rather than
being fully transferred to the rack investor.

The public record does not identify the Broadcom customer behind the $29 billion
backstop. The named Google/Anthropic TPU pathway therefore remains a separate edge
unless contract-level evidence connects it.

### Customer-contract and lease transmission

CoreWeave's two disclosed customer labels represented 45% and 20% of first-quarter
revenue. Its RPO rose to $98.8 billion, while named orders include up to approximately
$6.5 billion from OpenAI OpCo and an initial up-to-$14.2 billion from Meta. The filing
does not permit those names to be assigned to anonymized concentration labels.

The buildout also transmits risk to landlords and power systems. Beyond the $40.7
billion uncommenced-lease estimate, CoreWeave separately describes a 525 MW site with
$18.7–$19.6 billion of construction-cost-linked rent and other construction-linked
leases providing 363 MW. Whether the single-site range is included in the $40.7 billion
portfolio total remains unresolved, so it is not added.

## Meta Hyperion: outside capital with a $45.99 billion Meta exposure perimeter

Meta's Hyperion campus demonstrates why “off balance sheet” is not synonymous with
“risk transferred.” At the October 2025 closing:

- Meta contributed $4.3 billion of mostly construction-in-progress and land;
- a Blue Owl-managed fund affiliate contributed $7.0 billion in cash;
- the venture distributed $2.6 billion back to Meta; and
- ownership settled at 20% Meta and 80% Blue Owl affiliate.

By March 31, 2026, the parties had committed to fund their pro-rata shares of
approximately $27 billion in estimated development costs. Meta's equity investment
carrying value was $2.37 billion. Its leases begin in 2029 with an initial four-year,
$12.31 billion commitment and renewal options extending the total period to twenty
years.

Meta also supports terminal value. If it terminates or does not renew and campus fair
value falls below a declining threshold, Meta can owe the shortfall. That residual-value
threshold begins around $28 billion and decreases over time. No liability is recorded
because payment is not currently probable; the contractual channel still exists.

Meta reports $45.99 billion of maximum exposure, comprising the equity carrying value,
initial lease commitment, estimated future funding, and maximum residual-value
threshold. This total is the exposure perimeter, not a new amount to add to those four
components.

Transaction counsel adds one upstream capital layer that Meta's statements do not
quantify: a portion of the capital raised by Blue Owl-managed funds was financed
through private debt issued to PIMCO and selected bond investors. This establishes that
fixed-income capital sits behind at least part of the 80% venture interest. It does not
establish the debt principal, issuer, coupon, maturity, amortization, collateral, or
investor allocation. Public reports repeatedly use “$27 billion” for both development
and financing; the ledger refuses to equate those meanings without the private
placement or rating documents.

### Control and dependence point in different directions

Meta provides construction, administrative, and property-management services; is the
future tenant; holds 20%; and supplies the residual-value support. Yet Meta does not
consolidate the VIE because remarketing to future tenants and property-sale decisions
most significantly affect economic performance and Meta cannot direct those activities.

That is a clean example of the monograph's accounting-versus-economic-dependence rule:

```text
Blue Owl affiliate cash ────────► Hyperion venture assets
Meta land and construction ─────► Hyperion venture assets
Hyperion distribution ──────────► Meta
Meta future development funding ► Hyperion buildout
Hyperion future capacity ────────► Meta operations
Meta rent + residual support ────► Venture investors and lenders
PIMCO/other private debt ─────────► Blue Owl-managed project capital
Remarketing control ─────────────► outside Meta consolidation perimeter
```

The structure transfers most ownership and some development capital outside Meta, but
Meta remains the central demand, management, funding, and terminal-value node. A demand
shock can therefore hit Meta's equity investment, future funding decision, leased
capacity, and guarantee at once.

The model is not isolated. In March 2026 Meta classified another $1.48 billion of
data-center construction-in-progress and land as held for sale for expected contribution
to a third party to co-develop data centers. The next partner and whether the new
structure repeats Hyperion's lease and residual-value terms are not yet disclosed.

### Hyperion's real-estate VIE is not the whole Meta project perimeter

Louisiana regulatory records identify a separate legal layer. Meta told Entergy and the
Public Service Commission that Laidley LLC was the special-purpose entity developing
its original Richland Parish project, also known publicly as Project Sucre. Meta said
Entergy's ability to meet the project's energy schedule was critical to selecting the
site.

The original utility plan sought three combined-cycle plants totaling 2,260 MW,
substations, and transmission. Two approved Richland Parish plants totaling
approximately 1,500 MW began construction in December 2025 for late-2028 operation.
Entergy says Meta funds the full utility infrastructure needed to interconnect and
serve the data center.

The corporate and physical transition from Laidley to the later Blue Owl venture is
not publicly reconciled. Laidley may have held assets or rights that Meta later
contributed, or may retain a separate perimeter. Without contribution schedules and
utility assignment approvals, the original $10 billion project announcement and later
$27 billion Hyperion development estimate are potentially overlapping—not additive.

### The adjacent Evest project creates common-parent power concentration

July 2026 commission testimony identifies Evest LLC as a separate Meta subsidiary
proposing another hyperscale AI data center on land adjacent to Laidley. Its capacity
and most commercial terms are redacted. Entergy describes separate electric-service
and contribution agreements for Laidley and Evest, each supported by Meta.

The testimony makes the credit point explicit: despite two customer SPVs and two
agreement sets, a single Meta default could affect both. The legal separation does not
diversify parent credit.

Entergy's March 2026 public description of the expanded Meta relationship refers to
seven new gas combined-cycle plants totaling more than 5,200 MW, support for up to
2,500 MW of additional solar, Meta's full cost of service, and advertised customer
savings. Those portfolio figures cannot yet be allocated among Laidley, Evest, other
Meta expansion, and grid-wide need.

```text
Meta parent guarantee ──► Laidley utility obligations ──► original project power
                    └───► Evest utility obligations ─────► adjacent project power

Entergy generation + transmission ─► both customer SPVs
Meta load / payment performance ───► recovery of utility investment
```

This is a different risk channel from Hyperion's Blue Owl/PIMCO real-estate financing:
one supports land, buildings, and terminal value; the other supports generation,
transmission, interconnection, and regulated cost recovery. Both ultimately depend on
Meta's demand and credit.

## Microsoft and OpenAI: reciprocal economics without Microsoft control

### Microsoft's owned, leased, and still-uncommenced capacity

Microsoft spent $80.146 billion on additions to property and equipment during the nine
months ended March 31, 2026, including $30.876 billion in the March quarter. The
comparable prior-year amounts were $47.472 billion and $16.745 billion. This is a cash
flow, not a forecast or a liability.

Net property and equipment reached $283.228 billion at March 31, up from $204.966
billion at June 30, 2025. Within that total, gross servers, network equipment, and
software rose from $132.836 billion to $190.883 billion. Purchases of property and
equipment still sitting in accounts payable increased from $6.9 billion to $22.6
billion. The accounts-payable amount is unpaid acquisition activity that overlaps the
asset build; it cannot be added to cash capex as though it were an independent project.

Microsoft also recognized $22.238 billion of operating lease liabilities and $62.932
billion of finance lease liabilities. The more consequential forward boundary is
$196.6 billion of additional, primarily datacenter leases that had not commenced. Those
leases are scheduled to begin from fiscal 2026 through fiscal 2031 and run from one to
twenty-one years. They sit outside the recognized lease-liability table until
commencement.

This produces three distinct capital channels:

| Channel | March 31, 2026 measure | Accounting meaning |
|---|---:|---|
| Cash additions to property and equipment, nine months | $80.146B | cash already spent |
| Recognized finance and operating lease liabilities | $85.170B | present-value balance-sheet liabilities |
| Primarily datacenter leases not yet commenced | $196.6B | contracted future leases outside recognized liabilities |

The amounts are not additive measures of “AI spend.” They overlap different asset
populations and include infrastructure serving cloud, AI, and other corporate uses.
The ledger retains them separately.

### The OpenAI relationship has at least four economic channels

The October 2025 recapitalization and definitive agreement allow four channels to be
reconstructed without collapsing them into a generic strategic partnership:

1. Microsoft committed $13 billion of funding and had funded $11.8 billion by March 31,
   2026.
2. Microsoft held approximately 27% of OpenAI Group PBC on an as-converted basis. The
   October announcement valued that interest at approximately $135 billion, which is
   neither cash funding nor an audited carrying amount.
3. OpenAI contracted to purchase an incremental $250 billion of Azure services.
4. The companies retained revenue-sharing and intellectual-property arrangements.

Microsoft's $633 billion total remaining performance obligation at March 31, including
$627 billion commercial, is not allocated by customer. The public evidence does not
show how much of the $250 billion Azure contract is included in that balance, what
consumption schedule applies, or what pricing and remedies govern an underspend.

The April 2026 amendment changed the direction and exclusivity of the relationship.
Microsoft remains OpenAI's primary cloud partner and OpenAI products ship first on
Azure unless Microsoft cannot and chooses not to support the required capability.
However, OpenAI may serve all products across other cloud providers, and Microsoft's
model and product IP license through 2032 is now non-exclusive. Microsoft no longer
pays revenue share to OpenAI. OpenAI continues paying Microsoft through 2030 at the
same undisclosed percentage, independent of technology progress, subject to an
undisclosed aggregate cap.

### Economic ownership is not corporate control

The largest outside economic stake does not resolve control. OpenAI Foundation owns
26% of OpenAI Group PBC plus a valuation-linked warrant and holds exclusive special
voting and governance rights. It appoints every Group director and can replace them at
any time. Microsoft accounts for its approximately 27% interest under the equity method
and does not disclose control.

This creates a governance structure in which:

- Microsoft has a larger disclosed current economic percentage than the Foundation;
- the Foundation retains formal control;
- employees and other investors collectively hold the remaining 47%;
- Microsoft's commercial protection rests on contracts, IP rights, cloud priority,
  revenue share, and equity economics rather than board control.

That distinction matters in distress or strategic disagreement. Microsoft can be
economically exposed to OpenAI without possessing the authority to redirect OpenAI's
capital allocation or remove its directors.

### The accounting gain did not finance the build

Microsoft recorded $5.9 billion of net gains from OpenAI investments in the nine months
ended March 31, 2026, primarily from a dilution gain created by the recapitalization.
This is not evidence that OpenAI transferred $5.9 billion of cash to Microsoft and
should not be used as a funding source in the capital-flow reconstruction.

### Current interpretation

The evidence supports a reciprocal capital-and-demand system, not a closed circular
cash loop:

```text
Microsoft cash funding ───────────────► OpenAI Group equity capitalization
Microsoft datacenter/cloud capacity ──► OpenAI Azure consumption
OpenAI $250B purchase contract ───────► Microsoft customer-demand claim
OpenAI revenue share through 2030 ────► Microsoft contract economics
OpenAI operating and valuation result ─► Microsoft equity-method exposure
OpenAI Foundation governance ─────────► OpenAI Group strategic control
```

Calling this “circular revenue” would outrun the evidence. The $250 billion contract may
be supported by OpenAI operating cash, third-party equity, debt, customer revenue, or
other financing; its payment schedule is undisclosed. The correct finding is
reciprocity with unresolved funding provenance and execution risk.

## Amazon: customer capital embedded in the AWS demand system

### The infrastructure bridge

Amazon purchased $44.203 billion of property and equipment in the first quarter of
2026. Over the trailing twelve months it purchased $151.003 billion, or $147.299
billion net of property-sale proceeds and incentives. Net property and equipment
reached $397.458 billion.

The lease and commitment bridge is separately large:

| March 31, 2026 category | Amount | Measurement |
|---|---:|---|
| Operating lease liability | $91.617B | present value recognized |
| Finance lease liability | $13.325B | present value recognized |
| Leases not yet commenced | $106.347B | undiscounted future commitment |
| Unconditional purchase obligations | $103.768B | undiscounted off-balance-sheet minimums |
| Long-term debt face value | $122.632B | principal |
| AWS-focused long-term performance obligations | ~$364B | customer contract backlog |

Amazon's $569.280 billion contractual-commitment table is broader still, but includes
debt principal and interest, recognized lease payments, uncommenced leases, financing
obligations, purchase obligations, and other commitments. It is a payment schedule
across categories, not an AI exposure number.

Amazon issued $37 billion of dollar notes and €14.5 billion, or $16.782 billion
equivalent, of euro notes in March 2026. The issuance occurred in the same quarter as
the $44.203 billion property build and $15 billion OpenAI investment, but the notes
were issued for general corporate purposes. Timing alone does not prove that bond
investors financed either use.

### OpenAI: $50 billion of equity capital beside $138 billion of AWS demand

Amazon invested $15 billion in OpenAI Series C preferred stock during the first
quarter. A filed equity commitment obligates Amazon's investment subsidiary to
purchase an additional 50,895,254 shares for exactly $34,999,999,447.98. Amazon may
fund early at its discretion. Any remainder becomes mandatory upon a redacted
milestone or a public listing, subject to conditions, and the agreement expires by
December 31, 2028 unless accelerated.

The parent company does not merely provide informal support. Amazon.com, Inc.
irrevocably and unconditionally guarantees the investment vehicle's payment and
performance obligations. The guarantee is of payment and performance, not merely
collection.

At the same time, AWS and OpenAI expanded an existing $38 billion multi-year cloud
commitment by $100 billion over eight years. The contract includes obligations related
to the performance of AWS chips. The combined $138 billion relationship would equal
approximately 38% of Amazon's $364 billion AWS-focused performance obligations if all
of it were included at March 31. The filing does not provide the rollforward needed to
confirm that inclusion, so the ratio is only a scale comparison.

The filed equity agreement contains the most important unresolved connective tissue.
Closing is conditioned on redacted “Specified Event(s)” and redacted “Commercial
Agreements” provisions. The agreement expressly preserves rights and remedies under a
joint collaboration agreement, an AWS private-pricing addendum, and other commercial
arrangements. This proves legal cross-reference between the financing and commercial
relationship. It does not disclose whether AWS consumption or chip-performance
milestones trigger the $35 billion funding.

### Anthropic: compute delivery explicitly unlocks financing

The Anthropic structure is more explicit. Amazon invested $8 billion in convertible
notes from 2023 through 2025. At March 31, 2026, converted nonvoting preferred stock
had a $32.0 billion carrying value and remaining notes had a $42.2 billion fair value.
The $74.2 billion combined carrying amount reflects conversion and valuation, not
$74.2 billion of cash paid. During the quarter Amazon recorded a $4.5 billion
conversion-related gain and a $12.3 billion upward preferred-stock adjustment. Those
are noncash accounting changes.

After quarter end Amazon:

- invested another $5 billion in Anthropic nonvoting preferred stock;
- amended the AWS arrangement, including obligations tied to AWS-chip performance;
- created a financing facility of up to $20 billion;
- retained an option to invest up to $5 billion in future financings.

The option and facility are not additive: exercising the option reduces facility
availability dollar for dollar. More importantly, the facility starts with zero
drawable capacity. Availability is earned as Amazon achieves compute-capacity delivery
milestones. Anthropic may then draw by issuing new convertible notes, or common stock
after a liquidity event.

This contract allocates two-sided risk unusually clearly:

```text
Amazon delivers qualifying compute ──► Anthropic financing availability rises
Anthropic elects to draw ─────────────► Amazon supplies cash
Anthropic issues notes or stock ──────► Amazon receives investment asset
Anthropic consumes AWS capacity ──────► Amazon earns cloud revenue
```

Amazon therefore finances a customer only after Amazon has delivered the capacity
that makes the customer more able to consume Amazon's product. If Amazon fails to
deliver, Anthropic loses financing availability; if Anthropic fails economically,
Amazon faces cloud-demand, credit, and marked-investment exposure together.

### Power is already a long-duration financial contract

Amazon's energy contracts subject to derivative accounting cover approximately 200
million megawatt-hours with a weighted-average remaining term of about fifteen years;
most delivery is more than nine years away. When actual consumption is below committed
quantities, some contracts can settle through net cash rather than physical power.
The contracts are Level 3 instruments and can create cash settlements or termination
penalties. Their generators, regions, and AI allocation remain undisclosed.

### Current interpretation

Amazon now exhibits two different customer-capital designs:

| Customer | Supplier capital | Customer demand | Proven connective mechanism |
|---|---|---|---|
| OpenAI | $15B funded plus ~$35B guaranteed commitment | $138B AWS relationship | filed cross-references exist, but key commercial conditions are redacted |
| Anthropic | $13B funded through April 2026 plus up to $20B staged facility | AWS arrangement including AWS-chip obligations | compute-delivery milestones explicitly unlock financing |

Neither design proves sham demand or uneconomic purchasing. Both do mean that AWS
backlog, private-investment value, infrastructure utilization, and customer financing
cannot be treated as independent evidence of demand quality.

## Alphabet: the balance sheet now spans ownership, tenancy, credit support, and customer capital

Alphabet's first-quarter filing exposes a broader infrastructure-finance stack than
cash capex alone suggests. It spent $35.7 billion on property and equipment during the
quarter. Net property and equipment reached $281.020 billion, of which $108.597 billion
was not yet in service, and unpaid property purchases accrued in accounts payable were
$12.873 billion.

These measures describe different stages of one build:

| Measure at March 31, 2026 | Amount | Meaning |
|---|---:|---|
| Q1 capital expenditures | $35.700B | cash flow during the quarter |
| Net property and equipment | $281.020B | depreciated owned-asset stock |
| Assets not yet in service | $108.597B | construction/equipment embedded in the asset stock |
| Accrued property purchases | $12.873B | unpaid acquisition activity supporting the asset stock |

Approximately 60% of technical infrastructure was servers and networking equipment;
the rest was data-center land and buildings. The filing does not isolate which assets
serve Gemini, Cloud customers, Search, YouTube, or non-AI workloads. None of the four
table rows can be added to produce “total AI investment.”

### Leased capacity and contractual supply

Alphabet also recognized $16.161 billion of operating lease liabilities and $2.214
billion of finance lease liabilities. Beyond those present-value balances, it had
$75.6 billion of primarily data-center leases that had not commenced. Those leases
start from 2026 through 2031 and generally run from one to twenty-five years.

The wider contractual population was $332.4 billion. Within it, Alphabet disclosed
$232.7 billion of fixed or minimum long-term supply, energy, and content commitments.
Energy terms can extend through 2047 and generally contain take-or-pay provisions or
substantive termination fees. The $232.7 billion is a component of the broader
$332.4 billion population, not an additional amount. It also contains content and
non-AI supply, so it cannot yet be classified as an AI obligation.

### Third-party assets do not eliminate Alphabet's risk

Alphabet uses unconsolidated VIEs for some data-center leases, data-center credit
backstops, renewable-energy investments, and energy-infrastructure backstops. It says
it is not the primary beneficiary because it does not direct the activities that most
significantly affect those entities' economics. That accounting conclusion does not
remove the contractual transmission path.

At March 31, Alphabet disclosed:

- $28.4 billion maximum exposure under data-center credit derivatives;
- $9.0 billion maximum future payments under energy financial guarantees;
- up to $33.3 billion under a future backstop framework for data-center and
  energy-supply infrastructure, subject to final provider terms.

In April it executed approximately $15.3 billion of backstops with one unnamed
data-center provider against the $33.3 billion framework. That execution is not
incremental to the framework. It is also not part of the $37.4 billion existing
quarter-end backstop maximum because it occurred after the balance date.

The remedies are economically important. Following default Alphabet can assume
underlying leases for internal use, sublease capacity, or under specified conditions
pay to terminate an obligation. The risk can therefore migrate from a provider's
financing vehicle back to Alphabet as an operating asset, subleasing problem, or cash
settlement. A “not material” recorded guarantee fair value is not the same thing as a
small gross transmission channel.

```text
Third-party project owner / VIE
          │ owns or finances data-center or energy asset
          ▼
Alphabet lease, power contract, or backstop
          │ default can transfer use, sublease, or termination decision
          ▼
Alphabet operating capacity and contingent cash exposure
```

### River Bend exposes the project beneath one guarantee

Hut 8's River Bend financing makes a full third-party chain observable:

| Layer | Legal counterparty | Disclosed economics |
|---|---|---|
| Property / issuer | Hut 8 DC LLC | 245 MW IT capacity; 330 MW utility capacity |
| Project debt | private institutional noteholders | $3.25B; 6.192%; due 2042; amortization from 2028 |
| Parent sponsor | Hut 8 | owns issuer; project debt non-recourse to parent |
| Direct tenant | Fluidstack | 15-year triple-net lease; $7.0B base value |
| Credit backstop | Google | base rent plus related pass-through obligations |
| Model-company demand | Anthropic | infrastructure partnership; direct payment terms undisclosed |
| Construction | Hut 8 / Jacobs-linked contract | completion and guaranteed-maximum-price risk |

The bond closed on April 30, 2026. It has first-priority liens over substantially all
issuer assets plus a pledge of issuer equity. Proceeds finance construction, reimburse
an undisclosed amount of prior Hut 8 equity, establish debt-service reserves, and pay
transaction costs. The $3.25 billion principal is therefore not the total project cost.

Fluidstack's base lease includes 3% annual escalation. Three optional five-year
renewals would raise nominal contract value to $17.7 billion, and a right of first offer
covers up to another 1,000 MW if site power expands. Neither optional amount belongs in
base contracted capacity.

Google's guarantee changes the credit pathway without changing the tenant:

```text
Institutional bond cash ─► Hut 8 DC construction ─► River Bend capacity
River Bend capacity ─────► Fluidstack lease ──────► Anthropic compute pathway
Fluidstack rent default ─► Google guarantee ──────► Hut 8 DC / bond cash flow
Construction failure ────► delayed rent + bond risk before Google-backed operations
```

S&P's preliminary rating analysis confirms the separation. Google-supported lease cash
flow strengthened operations-phase credit, while the Jacobs-guaranteed design-build
contract constrained construction-phase risk. Investment-grade project debt is not a
claim that construction cannot fail; it is a judgment about the combined tenant,
guarantee, contractor, collateral, and structural protections.

The unresolved economic center is Anthropic. The public record links Anthropic and
Fluidstack to at least 245 MW and a wider development target of up to 2,295 MW, but
does not disclose Anthropic's minimum spend, remedies, or credit support. Bondholders
can therefore see Google-protected landlord cash flows more clearly than outsiders can
see the end customer's compute economics.

### Intersect moves one part of the system inside consolidation

Alphabet completed the Intersect acquisition on March 10 for $5.862 billion. The
purchase-price allocation included $5.111 billion of property and equipment and
$1.214 billion of assumed debt. Those components are not additive to consideration.
Alphabet determined that Intersect was a VIE and that Alphabet was now its primary
beneficiary, so it consolidated the entity.

The contrast matters. Intersect's projects and liabilities now sit inside Alphabet's
consolidated perimeter, while other leased and backstopped data-center and energy VIEs
remain outside it. The physical system cannot be reconstructed from consolidation
status alone; each project still requires an ownership, debt, lease, guarantee, power,
and capacity map.

### A $40 billion strategic investment beside TPU demand

Alphabet's filing disclosed an unnamed future private investment with two tranches:

1. $10.0 billion of capital expected to fund in the second quarter as a
   non-marketable security;
2. $30.0 billion of funding contingent on operational and financial milestones through
   2030 and accounted for as a Level 3 equity derivative.

Reuters separately identified Anthropic as the counterparty and reported that the
first tranche was at a $350 billion valuation, the second depended on Anthropic
performance targets, and the transaction supported compute expansion. The amounts and
timing match the filing, but the public investment agreement is unavailable. The
database therefore grades the amount and structure A while grading the identity and
valuation B.

Anthropic had separately announced a multi-gigawatt Google/Broadcom TPU agreement
starting in 2027. It also said Amazon remained its primary cloud and training partner.
This produces a correlated customer-capital pathway:

```text
Alphabet capital ───────────────► Anthropic equity / contingent funding
Google + Broadcom infrastructure ► Anthropic TPU capacity
Anthropic compute payments ─────► Google/Broadcom revenue and utilization
Anthropic performance ──────────► value of Alphabet's investment and later funding
```

The evidence does **not** show that TPU consumption triggers Alphabet's $30 billion
funding, that investment proceeds must be spent on Google capacity, or that cloud
payments recycle the invested cash. Those remain material unknowns rather than
assumptions.

### Debt and equity now explicitly fund the AI build

Alphabet's face amount of long-term debt rose to $80.305 billion after first-quarter
dollar, sterling, and Swiss-franc issuances produced about $31.1 billion of net
proceeds. The notes were issued for general corporate purposes, so the quarterly debt
cannot be mechanically assigned to AI.

By June the company made the broader source-of-funds logic explicit. It announced an
expected $80 billion equity package:

| Channel | Initial announced amount | Status and intended use |
|---|---:|---|
| Underwritten common stock | $15B | proposed; general corporate purposes including AI infrastructure |
| Mandatory-convertible preferred depositary shares | $15B | proposed; general corporate purposes including AI infrastructure and capped-call cost |
| Berkshire private placement | $10B | contracted at disclosed Class A and C prices |
| Common-stock ATM program | $40B | prospective from Q3; about $30B expected for employee-equity tax withholding |

The underwritten component was subsequently priced at approximately $35 billion,
raising the expected package to about $85 billion, but the ATM program remained
prospective. The headline amount is therefore neither all funded nor all infrastructure
capital.

Management paired the financing with $180–190 billion of expected 2026 capex,
$174 billion of trailing operating cash flow, and more than $85 billion of debt raised
over the prior year. It said AI demand exceeded available supply and that 2027 capex
would increase significantly again. This is direct evidence that Alphabet no longer
views operating cash flow alone as the preferred funding channel for its planned
compute trajectory.

### Current interpretation

Alphabet is not one node in the capital map. It is simultaneously:

- an owner and constructor of infrastructure;
- a long-duration tenant;
- a consolidating acquirer of power and data-center development;
- a credit supporter of third-party infrastructure vehicles;
- a power and supply offtaker;
- an unsecured debt issuer;
- a common and preferred equity issuer;
- a strategic investor in a major compute customer.

These roles increase funding capacity and control, but they concentrate several
failure modes in one balance sheet. A deterioration in Anthropic economics, cloud
demand, data-center delivery, power availability, or capital-market access can affect
more than one ledger at the same time.

## Oracle: customers finance GPUs while Oracle finances the facilities

Oracle's fiscal-year filing and supplemental capital bridge reveal a different
allocation of the AI build than the other hyperscalers. Fiscal 2026 capex was $55.663
billion, up 162%. Operating cash flow was $31.977 billion, leaving negative $23.686
billion of free cash flow under Oracle's definition.

That comparison still overstates Oracle's company-funded cash requirement. Oracle
reconciles GAAP capex to a lower “net cash outlay”:

| Fiscal 2026 capital channel | Amount | Classification |
|---|---:|---|
| GAAP capital expenditures | $55.663B | cash paid for capex |
| Less short-term manufacturer financing | $(3.345)B | financing cash flow tied to infrastructure production |
| Less customer prepayments with significant financing component | $(4.592)B | customer cash received before performance |
| Oracle supplemental net cash outlay | $47.726B | non-GAAP residual |

This is a genuine financing bridge, not an alternative estimate of capex. Oracle says
third-party manufacturers produce most cloud infrastructure assets and may receive
components Oracle bought directly. The customer prepayment is recorded in deferred
revenue and carries a significant financing component. The audited footnote rounds it
to $4.6 billion.

### The asset build is already larger than the prior year's entire PPE base

Net property and equipment rose from $43.522 billion to $99.957 billion. Construction
in progress reached $39.973 billion, primarily servers, networking equipment, and
leasehold improvements for data centers. Finance-lease ROU assets within PPE reached
$7.464 billion. These are components of the $99.957 billion balance, not additional
asset populations.

Oracle's lease stack is substantially larger:

| Lease channel at May 31, 2026 | Undiscounted / committed | Recognized liability |
|---|---:|---:|
| Operating leases | $41.867B | $30.190B |
| Finance leases | $11.460B | $7.701B |
| Additional substantially data-center leases not commenced | $260.000B | $0 before commencement |

The future leases generally begin from fiscal 2027 through fiscal 2029 and run fifteen
to nineteen years. One includes a guarantee of up to $3.3 billion of the lessor's
borrowing, maturing in September 2026. The guarantee is embedded in the lease
population rather than additive to it.

Oracle also had $13.309 billion of unconditional purchase and other obligations,
primarily data-center power. After year end it entered another $19 billion of
five-year cloud-infrastructure asset purchase commitments beginning in fiscal 2027.
This subsequent amount is not part of the May 31 power-obligation balance.

### $638 billion of backlog does not mean Oracle funds $638 billion of assets

Oracle's RPO increased from $138 billion to $638 billion in one year. The expected
recognition schedule is unusually long:

- 12% in the next twelve months;
- 34% in months 13–36;
- 34% in months 37–60;
- the remaining 20% thereafter.

The official earnings release adds a crucial ownership and financing fact. Most Q3 and
Q4 RPO growth came from large AI contracts where customers either prepaid Oracle to
purchase GPUs or bought and supplied the GPUs themselves. Oracle says the combined
prepaid and customer-supplied hardware portions total $75 billion.

That aggregate cannot be read as $75 billion of Oracle cash:

```text
Path A: Customer cash prepayment ─► Oracle buys GPUs ─► deferred revenue / financing component
Path B: Customer buys GPUs ───────► customer supplies hardware to Oracle ─► ownership terms unknown
Path C: Oracle debt/equity/cash ──► facilities, networking, power, and residual equipment
```

The filing and release do not split Path A from Path B or allocate the total among
customers. They also do not establish who owns customer-supplied hardware during or
after the cloud term, who bears refresh and obsolescence risk, or what happens after a
customer default.

Oracle named AMD, Meta, NVIDIA, OpenAI, TikTok, xAI, and others as the large contracted
OCI customers motivating its financing plan. It did not assign any named customer to
the $4.592 billion cash prepayment, $75 billion mixed hardware aggregate, $638 billion
RPO, or $260 billion lease population. The database refuses those tempting matches.

### Saline Township reveals one external project-capital chain

Related Digital's April 2026 announcement identifies a specific project behind part of
Oracle's physical expansion, but it does not permit a dollar match to Oracle's aggregate
lease disclosures:

| Layer | Disclosed role | Amount/status |
|---|---|---|
| Project | three-building Saline Township, Michigan campus | $16B announced project cost; >1 GW |
| Developer | Related Digital | construction underway |
| Sponsor capital | Related Digital and Blackstone-affiliated funds | equity amounts undisclosed |
| Project debt | PIMCO-managed funds and accounts as anchor | fixed-rate, long-term; amount and terms undisclosed |
| Cloud tenant | Oracle | purpose-built capacity; lease terms undisclosed |
| Model-company demand | OpenAI relationship | end-use link stated; contract allocation undisclosed |
| Utility | DTE | 100% of project power announced |
| Power capital | Oracle | new battery investment financed entirely by Oracle; amount undisclosed |

The developer said financing had been secured, while also saying closing remained
subject to customary conditions. The ledger therefore records announced/conditional
capital, not funded debt or equity.

This is a four-balance-sheet chain before even counting the utility or lenders:

```text
Related + Blackstone equity ──┐
PIMCO-anchored project debt ──┴─► Saline project assets
DTE resources + battery ─────────► project power
Oracle battery capital ──────────► power augmentation
Saline capacity ─────────────────► Oracle OCI
Oracle OCI services ─────────────► OpenAI demand
```

The $16 billion project cost is not Oracle capex, lease value, or OpenAI contract value.
Nor can it yet be assigned to the $260 billion uncommenced-lease balance. The missing
lease, loan, equity, and utility documents determine where construction, completion,
credit, and residual-value risk actually sit.

### Debt and equity close the residual funding gap

Oracle had $129.5 billion of outstanding indebtedness at year end. It raised about
$42.7 billion of senior-note proceeds during fiscal 2026 and had an undrawn $10
billion revolver. Oracle's earnings release summarizes total fiscal 2026 capital-market
funding as $43 billion of debt and $5 billion of equity.

For fiscal 2027, Oracle expects approximately $40 billion more debt and equity,
including its previously announced $20 billion ATM program. That planned raise follows
a February calendar-year financing plan that explicitly said the funds were needed to
build OCI capacity against contracted large-customer demand.

The reconstructed system is therefore:

```text
Operating cash ─────────────┐
Debt + preferred/common ────┼─► Oracle-owned equipment and construction
Manufacturer financing ─────┤
Customer prepayments ───────┘

Customer-purchased GPUs ─────────► Oracle-operated capacity (ownership unresolved)
Oracle long leases ──────────────► third-party data-center facilities
Oracle power commitments ────────► electricity supply
Customer cloud contracts ────────► $638B RPO and future utilization
```

### Current interpretation

Oracle has not eliminated capital intensity. It has divided it. Customers can fund or
supply GPUs, while Oracle retains a large share of facility, networking, power,
lease, integration, and execution risk. This reduces the immediate cash burden per
unit of contracted capacity but can amplify concentration: customer performance
supports RPO conversion, utilization, prepayment economics, and the justification for
long-lived lease and power commitments at the same time.

The maturity mismatch is also visible. AI accelerators may have useful lives of roughly
six years in Oracle's accounting, while new data-center leases run fifteen to nineteen
years and debt extends to 2066. The contracts may compensate Oracle for that mismatch,
but customer-level pricing, renewal, residual-value, and termination rights are not
public.

## OpenAI: the funding headline is a schedule, not a cash balance

The new `data/funding_timeline.csv` separates aggregate rounds, investor components,
tranches, superseded announcements, and debt capacity. That separation changes the
meaning of OpenAI's two largest funding headlines.

### The 2025 round is fully funded, but the first headline is obsolete

OpenAI announced $40 billion at a $300 billion post-money valuation in March 2025.
SoftBank's initial agreement contemplated up to $40 billion, up to $10 billion of
syndication, an April first close, and a conditional December second close. The
transaction ultimately changed:

| 2025 round component | Funded amount | Status |
|---|---:|---|
| SoftBank first closing | $7.5B | funded April 2025 |
| SoftBank second closing | $22.5B | funded December 26, 2025 |
| Third-party co-investors | $11.0B | funded |
| Final aggregate | $41.0B | fully funded |

The final $41 billion is not incremental to the announced $40 billion. It supersedes
it. SoftBank's $30 billion and the co-investors' $11 billion are components of the
$41 billion and therefore are also non-additive. SoftBank reported approximately 11%
ownership after the close.

The financing chain was already leveraged. SoftBank initially expected the first
closing to be financed through borrowing from Mizuho and other financial institutions.
The final source mix for each 2025 tranche is not fully disclosed, so the database
records the planned borrowing but does not label the entire $30 billion debt-funded.

### The 2026 round is closed, but most cash timing is not public

On February 27, OpenAI announced $110 billion of new investment at a $730 billion
pre-money valuation:

| Anchor | Announced allocation | Verified funded by July 25, 2026 |
|---|---:|---:|
| Amazon | $50B | $15B |
| SoftBank | $30B | $20B |
| NVIDIA | $30B | not separately disclosed |
| Anchor total | $110B | at least $35B |

OpenAI then announced on March 31 that it had closed the round with $122 billion in
**committed capital** at an $852 billion post-money valuation. The $122 billion
supersedes the $110 billion anchor announcement; it does not add to it. The arithmetic
$12 billion residual is a cohort, not a list of independently verified allocations.
OpenAI named Microsoft and many financial institutions and said more than $3 billion
came from individuals through bank channels, but did not publish their individual
amounts or aggregate cash receipts.

“Closed” and “committed” therefore cannot be translated into “$122 billion of cash on
OpenAI's balance sheet.” Counterparty filings establish at least two staged mechanisms:

- Amazon funded $15 billion and committed approximately $35 billion subject to
  specified conditions and an outside date.
- SoftBank contracted for three $10 billion tranches. The April and July tranches
  closed; the October tranche remained scheduled.

This creates a verified lower bound of $35 billion funded by those two 2026 anchors,
not an estimate of total cash received. Other investors may have funded substantial
amounts at close. NVIDIA's cash contribution remains unresolved.

### SoftBank transmits bank leverage directly into OpenAI equity

SoftBank arranged a $40 billion unsecured bridge facility from JPMorgan, Goldman
Sachs, Mizuho, SMBC, and MUFG. It matures March 25, 2027 and supports the OpenAI
follow-on investment plus general corporate purposes. On July 1 SoftBank drew exactly
$10 billion under that bridge and used it to fund the second OpenAI tranche the same
day.

The documented path is:

```text
Five-bank bridge syndicate ──$10B draw──► SoftBank / SVF2
                                           │
                                           └──$10B preferred-share tranche──► OpenAI
```

This is not indirect macroeconomic attribution. It is a specifically disclosed
source-and-use transfer from short-dated bank credit into private OpenAI equity.
SoftBank expects to refinance or repay the bridge through existing assets and other
financing measures. The eventual loss-bearing path therefore can extend from OpenAI's
private valuation into SoftBank NAV, asset monetization, and its bank lenders.

### NVIDIA's two announcements cannot be combined

The September 2025 NVIDIA/OpenAI announcement was a letter of intent for at least ten
gigawatts of NVIDIA systems and an NVIDIA intention to invest up to $100 billion
progressively as each gigawatt deployed. The parties said details still had to be
finalized.

OpenAI's February 2026 announcement instead allocated $30 billion of its new round to
NVIDIA and paired the relationship with 3 GW of dedicated inference capacity and 2 GW
of Vera Rubin training. The reviewed sources do not say that $30 billion is the first
tranche of the earlier $100 billion framework.

NVIDIA's April 26 filing disclosed $27 billion of total investment commitments subject
to contingencies, expected through the rest of fiscal 2027. It did not allocate that
aggregate to OpenAI. The apparent $3 billion difference cannot be interpreted as a
funded amount: the aggregate may include other investments, exclude amounts already
funded, or reflect conditions and accounting boundaries not publicly reconciled.

### Stargate is a program boundary, not a $500 billion funded entity

The January 2025 announcement called Stargate a new company that “intends” to invest
$500 billion over four years and begin deploying $100 billion immediately. It named
SoftBank, OpenAI, Oracle, and MGX as initial equity funders; assigned SoftBank financial
responsibility and OpenAI operational responsibility; and simultaneously said
definitive agreements were still being finalized.

The ledger therefore records $500 billion as an announced program intention with zero
verified subscribed or funded capital—not as an obligation, asset, or cash balance.
The economic system must instead be reconstructed from the contracts and projects that
later implement pieces of Stargate.

The SB Energy transaction supplies one such piece and exposes a multi-directional
pathway:

```text
OpenAI ──$500M announced equity──► SB Energy ◄──$500M announced equity── SoftBank
                                      ▲
Ares ──$800M redeemable preferred────┘

OpenAI ──1.2 GW lease payments──► SB Energy
SB Energy ──API / ChatGPT spend──► OpenAI
```

SB Energy is also the developer and operator selected for OpenAI's Milam County site.
This is genuine reciprocal capital and demand, but not proof of artificial revenue or
round-tripped cash. The investment closing status, lease dollars, project cost, power
economics, preferred-equity terms, and SB Energy's OpenAI spend are all undisclosed.

### Current interpretation

OpenAI's financing capacity is real but maturity- and condition-dependent. The
headline round totals mix cash already received, scheduled equity tranches, conditional
equity, unresolved investor funding, and an undrawn $4.7 billion revolver. OpenAI also
holds long-duration compute and capacity relationships across Microsoft, AWS, Oracle,
CoreWeave, Google, NVIDIA, AMD, Broadcom, and SB Energy.

That mismatch is the central forensic question: not whether OpenAI “raised $122
billion,” but whether the timing and conditions of actual liquidity remain aligned
with compute payments, lease starts, power delivery, and infrastructure completion.

## Anthropic: capital is explicitly unlocked by compute delivery

Anthropic's company-confirmed chronology now spans eight disclosed funding events plus
an unresolved 2024 round:

| Date | Funding event | Amount | Post-money valuation | Treatment |
|---|---|---:|---:|---|
| May 2021 | Series A | $124M | undisclosed | funded |
| April 2022 | Series B | $580M | undisclosed | funded |
| May 2023 | Series C | $450M | undisclosed | funded |
| September 2023–March 2024 | Amazon initial strategic investment | $4.0B | undisclosed | funded in $1.25B + $2.75B tranches |
| November 2024 | Amazon incremental strategic investment | $4.0B | undisclosed | funded by 2025 |
| March 2025 | Series E | $3.5B | $61.5B | raised |
| September 2025 | Series F | $13.0B | $183B | raised |
| February 2026 | Series G | $30.0B | $380B | raised |
| May 2026 | Series H | $65.0B | $965B | raised; embeds prior hyperscaler commitments |

These company-confirmed rows represent at least $120.654 billion of disclosed funding.
That is not a complete lifetime total: it excludes an unresolved Menlo-led 2024
financing, earlier Google strategic capital that has not yet been reconstructed from
primary documents, and contingent future facilities. It also does not add the
components embedded within Series H.

The acceleration is concentrated. Series G and H supplied $95 billion in roughly
three and a half months, compared with $25.654 billion across the company-confirmed
rows listed before February 2026. Valuation rose from $183 billion after Series F to
$965 billion after Series H in less than nine months. Those are financing marks, not
audited enterprise-value appraisals or realizable liquidation values.

### Series H incorporates prior commitments

Anthropic explicitly said the $65 billion Series H included $15 billion of previously
committed hyperscaler investments, including $5 billion from Amazon. Therefore:

```text
Series H aggregate                         $65B
  embedded hyperscaler commitments        $15B
  implied other Series H capital           $50B
```

The $50 billion is arithmetic, not a disclosed investor schedule. The remaining
$10 billion of the hyperscaler subset matches Alphabet's contemporaneous initial
$10 billion Anthropic-linked security by amount, timing, and reported counterparty.
That is strong evidence, but no primary source explicitly links the two. The database
marks it Grade C and leaves Alphabet's separate $30 billion performance-contingent
funding outside Series H.

This prevents three errors:

1. adding $65 billion of Series H to the embedded $15 billion;
2. adding Amazon's $5 billion April investment again;
3. treating Alphabet's full $40 billion program as funded Series H cash.

### Amazon's financing availability is earned by Amazon's own delivery

The April 2026 expansion is a more explicit capital-contract loop than the OpenAI
arrangements currently disclose. Anthropic committed more than $100 billion over ten
years to AWS technologies and secured up to 5 GW across Graviton and Trainium
generations. It reported already using more than one million Trainium2 chips.

Amazon invested $5 billion, taking its cumulative cash investment to $13 billion, and
created a facility of up to $20 billion. Amazon's filing states that facility
availability increases when **Amazon achieves compute-capacity delivery milestones**.
Draws would purchase new convertible notes or, after a listing, common stock. A
separate option for up to $5 billion is not an additional facility: exercising it
reduces availability under the $20 billion.

The causality is therefore:

```text
Amazon builds and delivers AWS capacity
              │
              ├──► Anthropic can consume contracted compute
              └──► Anthropic unlocks more Amazon financing availability

Amazon capital ──► Anthropic ──cloud payments──► AWS
Amazon chips   ──► Anthropic ──model access────► Bedrock customers
```

This does not prove that Amazon capital is spent dollar-for-dollar back at AWS.
Facility availability, cloud payments, investment cash, and AWS revenue recognition
occur on different schedules. It does prove that infrastructure delivery and funding
availability are contractually coupled.

Amazon simultaneously holds:

- $13 billion of historical cash investment;
- a future facility up to $20 billion;
- noncash private-investment carrying values that had risen well above cash cost;
- a customer commitment greater than $100 billion;
- primary-cloud and primary-training status;
- chip co-design and Bedrock distribution economics.

Those exposures cannot be summed into loss-at-default. They can, however, fail
together. Anthropic weakness can reduce AWS utilization, financing recoverability,
private-investment value, and Bedrock differentiation. Amazon delivery weakness can
withhold both compute and finance from Anthropic.

### Revenue growth is relevant, but not yet a coverage ratio

Anthropic reported annualized run-rate revenue of about $1 billion at the start of
2025, more than $5 billion in August 2025, more than $30 billion by April 2026, and
more than $47 billion in May. These are management run rates rather than audited
revenue, cash receipts, or free cash flow.

The greater-than-$100-billion AWS commitment would average greater than $10 billion a
year if spread evenly, but the actual ramp, pricing, take-or-pay floor, gross-versus-net
marketplace accounting, and termination terms are undisclosed. Comparing a $47
billion exit run rate directly with a hypothetical $10 billion annual cloud cost would
create false coverage precision.

### Multicloud diversification does not remove concentration

Anthropic separately contracted for multiple gigawatts of next-generation Google and
Broadcom TPU capacity beginning in 2027, including approximately 3.5 GW described in
Broadcom's filing. It says Claude is available across AWS, Google Cloud, and Microsoft
Azure. Yet Anthropic still describes Amazon as its primary cloud and training partner.

The diversification changes substitution options but creates a larger coordination
problem: Anthropic must fund, schedule, and utilize capacity across different chips,
clouds, and contract structures while financing milestones themselves depend on one
provider's delivery.

### Current interpretation

Anthropic has the strongest documented example so far of a hyperscaler serving as
investor, lender, chip supplier, primary cloud, training partner, distributor, and
contract beneficiary at once. The relationship can accelerate scaling because capital
and capacity arrive together. It can also transmit one operational failure through
multiple accounting channels.

The immediate unresolved question is not simply whether Anthropic can “afford” its
compute. It is whether the timing of recognized revenue, cash collection, Series H
proceeds, conditional Amazon draws, and actual multi-cloud delivery remains ahead of
the payment and utilization curve.

### Anthropic added a third major compute pathway—and its supplier is a competitor

SpaceX disclosed a May 2026 cloud-services agreement under which Anthropic receives
access to approximately 325,000 NVIDIA GPUs across Colossus I and II. The schedule is
$1.25 billion per month through May 2029 after reduced ramp fees in May and June 2026.
Anthropic retains its models, content, data, and related intellectual property.

The headline arithmetic—$1.25 billion multiplied by 36 months—is approximately $45
billion. That is a schedule ceiling, not fixed backlog. After the first three months,
either party can terminate on 90 days' notice. Public terms do not disclose the reduced
ramp fees, service credits, minimum payment during the notice period, or capacity
remedies. The enforceable floor therefore cannot be calculated.

This relationship is economically important for two reasons:

1. Anthropic now has a disclosed NVIDIA-compute pathway through a direct frontier-model
   competitor in addition to its AWS/Trainium and Google/Broadcom TPU pathways.
2. xAI can monetize infrastructure externally rather than relying only on Grok and X
   demand. The same capacity is therefore both a strategic input and a merchant asset.

The contract reduces single-provider hardware dependence but introduces a short-exit
counterparty risk. Anthropic can leave; SpaceX can reclaim capacity; and both parties
retain incentives to terminate if internal model demand becomes more valuable than the
contract.

## xAI and SpaceXAI: equity funded the sprint; affiliated leases financed the hardware

The xAI reconstruction changes the unit of analysis. xAI is no longer an isolated
private lab. It merged with X in March 2025 and then entered SpaceX in February 2026
through a common-control transaction. SpaceX retrospectively recast prior financial
statements and reports X and former xAI together as one AI segment. The filed segment
numbers are therefore real but not stand-alone xAI results.

### Funding chronology

| Date | Event | Amount | Status | Additive treatment |
|---|---:|---:|---|---|
| May 2024 | Series B | $6B | company says raised | additive |
| Dec. 2024 | Series C | $6B | company says closed | additive |
| June 2025 | two term loans plus secured notes | $5B | funded debt; later refinanced | additive financing source |
| Jan. 2026 | Series E | $20B | company says completed | additive aggregate |
| March 2026 | Tesla component | $2B | funded as SpaceX Class A after conversion | embedded in Series E |

The company-confirmed equity chronology therefore contains at least $32 billion in
Series B, C, and E funding. It does not yet include a verified Series A or Series D.
The $2 billion Tesla investment must not be added to $20 billion Series E: Tesla
originally contracted for xAI preferred stock, but regulatory timing pushed closing
past the SpaceX merger, so its right converted into SpaceX Class A shares.

### The merger was a reorganization—not an arm's-length purchase-price signal

SpaceX closed the two-step merger on February 2, 2026, issuing 321.7 million Class A
shares and 121.7 million Class B shares on a pre-split basis and paying $2.947 billion
cash to xAI common and preferred holders. The legal exchange ratio and cash election
are documented, but common-control accounting means the resulting financial statements
were retrospectively combined at carrying values.

This distinction blocks two common errors:

- the issued shares are not a clean arm's-length fair-value price for xAI; and
- the recast AI segment cannot be read as the organic history of one business.

The segment includes X advertising, subscriptions, data licensing, Grok, and compute
infrastructure. Legal-entity debt and related-party footnotes are needed to recover the
underlying xAI capital pathways.

### The operating scale is large and the coverage gap is larger

The combined AI segment reported:

| Measure | 2025 | Q1 2026 |
|---|---:|---:|
| revenue | $3.201B | $0.818B |
| operating loss | $(6.355)B | $(2.469)B |
| depreciation and amortization | $3.568B | $1.493B |
| capital expenditures | $12.727B | $7.723B |

These are GAAP segment disclosures, but they combine X and xAI. They nevertheless show
the funding problem's order of magnitude: first-quarter AI capex alone was more than
nine times segment revenue, before operating losses. The evidence supports a
capital-consumption conclusion; it does not support a pure-xAI burn rate.

### xAI first borrowed at private-credit-like rates

In June 2025, xAI raised:

- a $1 billion fixed-rate first-lien term loan at 12.5%;
- a $1 billion floating first-lien term loan at SOFR plus 7.25%; and
- $3 billion of 12.5% senior secured notes.

All were due in 2030 and supported by guarantees from xAI subsidiaries. SpaceX repaid
the two loans on March 2, 2026 and the notes on March 5. The disclosed prepayment
penalties on the xAI instruments totaled $739 million. The wider refinancing produced
a $1.526 billion group debt-extinguishment loss.

SpaceX replaced this stack with a $20 billion unsecured bridge facility bearing SOFR
plus 0.75% to 1.75%, subject to rating, and jointly guaranteed by X Corp., X.AI LLC,
and CTC Property. The bridge also repaid legacy X debt, so it cannot be attributed
entirely to xAI. Economically, however, the transaction moved a costly, subsidiary-level
AI debt layer into the parent capital structure and cross-linked Space, Connectivity,
X, and xAI credit.

### The compute “SPV” is three affiliated lease financings

The filed structure is more precise than the reported shorthand. CTC, an xAI
subsidiary, entered three equipment leases with Valor Equity Partners, whose founder
serves on the SpaceX board:

| Transaction | Date | Undiscounted payments | Filed accounting |
|---|---|---:|---|
| Valor I | Oct./Nov. 2025 | $6.986B | failed sale-leaseback |
| Valor II | Jan./Feb. 2026 | $6.633B | failed sale-leaseback |
| Valor III | April 2026 | $6.587B | five-year equipment lease; subsequent-event classification unresolved |
| **Total** |  | **$20.206B** | mixed |

Valor I and II left $9.041 billion of recognized debt at March 31, 2026. SpaceX or a
subsidiary guarantees lessee payment and performance obligations across the disclosed
arrangements. In 2025 the group paid $885 million under the leases; in the first four
months of 2026 it paid $1.917 billion.

This is the core forensic discovery. The compute assets did not disappear from economic
risk because they sat in lease structures. Failed sale-leaseback accounting kept the
first two asset and debt positions on the consolidated balance sheet. The related-party
lessor provided a financing channel; group guarantees transmitted performance risk to
the post-merger parent.

Related-party status is not evidence of bad pricing or wrongdoing. It is evidence that
governance, capital supply, equipment ownership, and board oversight intersect in the
same pathway. The still-missing layer is Valor's own funding: lenders, SPVs, GPU title,
residual value, collateral, and enforcement rights.

### Tesla sits on both sides of the loop

Tesla funded $2 billion of SpaceX equity converted from its xAI Series E commitment.
xAI and SpaceX also purchased at least $506 million of Tesla Megapacks in 2025 and
another $303 million of Tesla goods through April 2026. xAI recognized only $3 million
of revenue from Tesla over 2025 and the first four months of 2026.

Capital and equipment therefore move in opposite directions:

```text
Tesla cash ───────────────► SpaceX equity
Tesla Megapacks/services ─► xAI / SpaceX AI infrastructure
xAI / SpaceX cash ────────► Tesla equipment revenue
xAI services ─────────────► Tesla vehicles and operations
```

This is a documented reciprocal relationship, but it is not a wash. Timing, margins,
asset life, ownership percentage, and service value differ. The correct interpretation
is related-party capital and supply concentration, not circular revenue by default.

### External compute converts an internal asset into a merchant business

The Anthropic agreement covers approximately 325,000 NVIDIA GPUs and schedules $1.25
billion monthly payments after ramp. That annualized rate—$15 billion—is several times
the combined AI segment's 2025 revenue, but the contract is terminable and had not yet
generated a comparable reported period.

This gives SpaceX two competing allocation choices:

```text
Colossus capacity ──► Grok training and inference ──► proprietary model value
                  └─► Anthropic cloud service ─────► external cash revenue
```

The optionality is genuine. So is the conflict: a rise in internal Grok demand can
increase the opportunity cost of serving Anthropic, while Anthropic termination can
strand merchant capacity unless SpaceX can redeploy it internally or to another
customer.

### Power is now a balance-sheet and permitting pathway

SpaceX disclosed an $805 million turbine purchase commitment through 2029 and a
separate approximately $2 billion mobile-turbine asset acquisition for AI data-center
power. It also disclosed litigation alleging that Colossus II turbines lacked required
Clean Air Act permits. The litigation outcome is unresolved.

Power risk is therefore not an abstract utility constraint. It connects equipment
purchase commitments, owned generation assets, environmental permits, data-center
uptime, and the ability to satisfy external compute contracts.

### Current interpretation

xAI's scaling was financed by at least four distinct channels:

1. $32 billion of company-confirmed Series B, C, and E equity;
2. $5 billion of high-cost secured debt in 2025;
3. $20.206 billion of affiliated equipment-lease payment schedules; and
4. post-merger access to SpaceX's $20 billion parent bridge and broader cash flows.

Those figures must not be summed as “capital raised.” They include different dates,
stocks, schedules, embedded components, and refinancing. Together they show the actual
mechanism: equity financed enterprise risk, secured debt accelerated the initial build,
affiliated leases financed compute hardware, and the common-control merger shifted
refinancing and guarantee capacity to SpaceX.

The xAI/SpaceXAI case also changes the ecosystem map. Anthropic is not only an Amazon
and Google compute customer; it is now a major terminable customer of its competitor's
infrastructure. Tesla is not only an equipment vendor; it is an equity holder. Valor
is not only an investor; it is an affiliated compute lessor. SpaceX is not only the
owner; its other operating businesses now share the refinancing and guarantee
perimeter.

## Mistral: an industrial customer became the anchor strategic investor

Mistral's available chronology is smaller and less completely disclosed than the U.S.
labs. The company confirms the dates of its seed, Series A, and Series B but does not
publish the amounts on its current history page. Reuters reported a €600 million
Series B in June 2024 at a €5.8 billion valuation; accessible reporting indicates it
mixed equity and debt, so the database does not classify the full amount as equity.

The primary-evidence anchor is the September 2025 Series C:

| Item | Amount / right | Evidence status |
|---|---:|---|
| Series C aggregate | €1.7B | company-confirmed |
| post-money valuation | €11.7B | company-confirmed |
| ASML investment | €1.3B | both counterparties confirmed |
| ASML fully diluted ownership | approximately 11% | ASML-confirmed |
| ASML governance | Strategic Committee advisory seat | both counterparties confirmed |
| other-investor residual | €0.4B | arithmetic residual |

ASML supplied 76% of the disclosed Series C. Its role is not merely financial. The
parties established a long-term collaboration to use Mistral models across ASML's
products, research, development, and operations. The committee seat gives ASML
advisory influence without disclosed control.

This is an industrial customer-investor loop:

```text
ASML cash ───────────────► Mistral equity
Mistral AI systems ──────► ASML products / R&D / operations
Mistral performance ─────► value of ASML's approximately 11% stake
ASML strategy ───────────► Mistral industrial demand and advisory influence
```

It is not a direct hardware-financing loop. ASML supplies lithography equipment to
chip manufacturers upstream; public evidence does not show Mistral buying ASML
machines. Treating ASML as Mistral's GPU vendor would skip the foundry, accelerator,
server, and cloud layers.

### Mistral is vertically integrating from models into merchant compute

Mistral Compute launched in June 2025 as an integrated GPU, orchestration, API, and
managed-platform business. The company initially advertised tens of thousands of
NVIDIA GPUs. Its current timeline says:

- procurement began in April 2025;
- GB200 racks arrived in July 2025;
- GB300 arrived and the GB200 alpha stack went live in December;
- GB200 served production and a Sweden site entered development in February 2026;
- first external customers onboarded in March 2026; and
- the target is 200 MW of sovereign EU capacity by 2027.

The important distinction is between operational milestones and the 200 MW target.
Mistral does not disclose current operating megawatts, GPU ownership, lease obligations,
project debt, customer minimums, or funded capex. The target therefore belongs in the
capacity ledger—not the liability ledger.

Mistral's economic perimeter now spans:

```text
private equity ─► model research ─► API / enterprise products
                                  └► Mistral Compute procurement
NVIDIA systems ───────────────────► internal training + external GPU cloud
industrial customers ─────────────► model revenue + merchant-compute utilization
```

That structure can lower dependence on U.S. hyperscalers and create external
infrastructure revenue. It also converts a relatively asset-light model company into a
capacity owner or long-term capacity intermediary. The unresolved financing and asset
ownership behind the 200 MW plan is therefore more material than the marketing label
“sovereign cloud.”

## Cohere: public capital supports a third-party facility while a future investor
would also supply the cloud

Cohere's confirmed 2025 round closed in two pieces: $500 million in August at a $6.8
billion valuation and another $100 million in September. High-quality reporting places
an earlier 2024 round at $500 million and a $5.5 billion valuation. The investor-level
allocations and security rights remain undisclosed.

### The Canadian compute project separates owner, tenant, and public funder

Canada finalized an up-to-C$240 million contribution to a C$725 million domestic
compute project. A later government assessment identifies the structure:

- CoreWeave builds and operates the Cambridge, Ontario facility;
- Cohere acts as anchor tenant; and
- the government contribution supports the project rather than buying Cohere equity.

This separates three capital roles:

```text
Government of Canada contribution ─► eligible compute project costs
CoreWeave / project capital ────────► facility ownership and operation
Cohere customer obligation ─────────► anchor utilization and project revenue
Compute capacity ────────────────────► Cohere models and other Canadian users
```

The public amount is a ceiling, not proof of C$240 million disbursed. The C$725 million
project value is not Cohere capex and not Cohere's contract value. The missing C$485
million cannot be labeled CoreWeave equity or debt without project documents; it may
include multiple private sources and customer economics.

This is the first lab case in the current research pass where government capital
explicitly supports a third-party compute owner while the model company anchors demand.
It creates a four-way dependency among subsidy conditions, facility financing,
CoreWeave execution, and Cohere tenancy.

### The planned Cohere/Aleph/Schwarz structure is not yet funded

In April 2026, Cohere and Aleph Alpha announced that they intended to join forces.
Schwarz Group companies said they intended to lead an upcoming Series E with €500
million—described as approximately $600 million—of structured financing. The planned
combined offering would deploy on Schwarz's STACKIT sovereign cloud.

The announcement uses future tense for both the combination and financing. It does not
establish a closed merger, funded capital, security type, minimum STACKIT spend, or
exclusivity. Those entries remain `announced_intent`.

If completed, Schwarz would occupy three positions:

1. lead financier of the combined model company;
2. infrastructure provider through STACKIT; and
3. distribution/deployment partner for sovereign enterprise customers.

This resembles the Amazon-Anthropic capital-contract loop in topology but not in
evidence maturity. Amazon's facility and compute milestones are filed agreements;
Schwarz's proposed structure is still an announcement. Conflating them would turn a
future plan into a current liability.

### Current interpretation

Cohere is pursuing sovereignty through two opposite ownership models at once:

- in Canada, it anchors a CoreWeave-owned facility supported by public capital; and
- in Europe, it proposes to pair strategic financing with a customer-facing cloud
  controlled by Schwarz.

Sovereignty therefore does not mean the model company owns the data center. It can mean
domestic location, public subsidy, local cloud control, contractual data governance, or
some combination. The financial risk holder differs in each version.

## Common bridge: comparability is now an explicit data property

The first cross-company bridge is frozen in `data/capital_bridge.csv`. Its 25 rows
cover Meta, Microsoft, Alphabet, Amazon, and Oracle and classify every amount by:

- temporal basis — quarter, nine months, annual, forecast, or point-in-time;
- measurement class — cash flow, recognized liability stock, commitment stock,
  contingent maximum, or mixed asset/cash support;
- recognition state — spent, recognized, contracted, contingent, or announced;
- AI attribution — dedicated, primarily data center, mixed infrastructure, or
  untraceable general corporate funding;
- overlap group and additive eligibility.

Eight of the 25 rows are deliberately non-additive. Examples include Meta's aggregate
commitments beside its likely overlapping lease commitments, Alphabet's broad
contractual-obligation total beside component categories, Oracle's derived net cash
outlay beside its capex and financing offsets, and Oracle's $75 billion mixed
customer-funding aggregate.

“Additive eligible” does not mean the 17 remaining rows can be summed into an industry
total. A quarterly cash flow, a nine-month cash flow, annual capex, a debt stock, and a
future lease stock remain different dimensions. The flag means only that the row is
not known to duplicate another row within its defined entity-period overlap group.

The bridge establishes three preliminary cross-company facts:

1. All five anchor companies use owned and leased infrastructure together; cash capex
   is never the full asset-acquisition channel.
2. Microsoft and Oracle disclose the largest clearly data-center-weighted uncommenced
   lease populations in the current bridge—$196.6 billion and $260 billion
   respectively—but commencement windows, asset ownership, and customer backing
   differ.
3. Oracle is the only anchor so far to quantify a customer/manufacturer funding offset
   directly against reported capex. That makes its $47.726 billion net cash outlay
   economically informative but not GAAP-comparable with peers' gross capex.

## Pass 1G — frozen capital-flow and dependency graphs

Chapter 2's graph inputs are frozen under an explicit overlap policy. The policy selects
investor components when they reconcile to a final round total and otherwise retains
the aggregate. It excludes superseded announcements, overlapping tranche detail, and
embedded investment options. The generator fails if the selected components no longer
reconcile for the OpenAI 2025 and 2026 rounds, Anthropic Series H, Mistral Series C,
xAI Series E, or Cohere's 2025 round.

The resulting capital map deliberately has two measures:

1. **selected committed capital** — enforceable, closed-with-commitment, or otherwise
   disclosed capital availability under the event's recorded status; and
2. **verified funded floor** — only amounts for which the reviewed evidence establishes
   funding.

These are cumulative program histories, not present cash balances. They mix equity,
strategic financing, and explicitly labeled credit facilities but exclude valuations,
revenue contracts, capex, guarantees, and lease notional. Capital could already have
been spent.

| Recipient | Currency | selected committed | verified funded floor | unverified or future gap |
|---|---:|---:|---:|---:|
| OpenAI Group | USD | $180.700B | $87.800B | $92.900B |
| Anthropic | USD | $170.654B | $110.654B | $60.000B |
| xAI | USD | $32.000B | $32.000B | — |
| SB Energy | USD | $1.800B | — | $1.800B |
| Cohere | USD | $1.700B | $1.100B | $0.600B |
| Mistral | EUR | €2.300B | €1.700B | €0.600B |
| Cohere public compute support | CAD | C$0.240B | — | C$0.240B |

OpenAI's committed total includes the selected historical Microsoft program, final
2025 round components, final 2026 round components, and the undrawn revolver. Its
$87.8 billion funded figure is a lower bound, not a claim that only that amount was
received: several 2026 investor components lack separately verified funded amounts.

Anthropic's $60 billion gap is reconstructible: $10 billion within Series H lacks
verified funding in the reviewed evidence, while Amazon's future $20 billion facility
and Alphabet's $30 billion contingent program remained future or milestone-dependent.

xAI's zero gap means the selected Series B, C, and E round aggregates were reported
raised. It does not mean $32 billion remained liquid after Colossus construction, debt
repayment, the SpaceX combination, or other uses.

### Dependency graph boundary

The separate dependency graph contains 126 typed edges:

| Edge source | Count |
|---|---:|
| operating and counterparty dependencies | 51 |
| contracts | 35 |
| ownership and control | 14 |
| guarantees and credit support | 12 |
| compute supply | 10 |
| power agreements | 4 |

This graph intentionally does not assign dollar flow to every edge. A guarantee,
ownership percentage, gigawatt commitment, supplier concentration, and customer
contract remain different measures. Unnamed counterparties are represented as explicit
unresolved cohorts so material guarantees do not disappear merely because the legal
beneficiary is not public.

Frozen artifacts:

- `reports/ai_financial_system/chapters/chapter_02/data/capital_edges_all_bases.csv`
- `reports/ai_financial_system/chapters/chapter_02/data/capital_sankey_committed.csv`
- `reports/ai_financial_system/chapters/chapter_02/data/capital_sankey_funded_cash.csv`
- `reports/ai_financial_system/chapters/chapter_02/data/capital_summary_by_recipient.csv`
- `reports/ai_financial_system/chapters/chapter_02/data/dependency_graph_edges.csv`
- `reports/ai_financial_system/chapters/chapter_02/data/graph_manifest.json`

## First discovery

Meta alone demonstrates why headline aggregation fails. The disclosed values measure at
least five different things: an annual spending forecast, a stock of non-cancelable
commitments, future leases, a reducible contingent capacity obligation, and restricted
cash. A naive sum would mix flows and stocks, duplicate overlapping obligations, combine
different dates, and treat an upper bound as a fixed amount.

This is not merely an accounting presentation issue. Each quantity points to a
different risk holder and transmission mechanism:

- capex guidance exposes Meta to execution and cash-flow risk;
- contractual commitments expose Meta and undisclosed infrastructure suppliers to
  cancellation and demand-realization risk;
- uncommenced leases expose future fixed-cost absorption and landlord/project-finance
  chains;
- contingent capacity creates optionality shared with a provider that may remarket it;
- restricted cash reveals contractual support that reduces Meta's freely available
  liquidity while protecting a counterparty.

## Research-pass disposition

The initial extraction list is complete at the level necessary for Chapter 2:

1. public-company capital bridges now cover the anchor hyperscalers plus NVIDIA, AMD,
   Broadcom, and CoreWeave;
2. quarter-over-quarter commitment changes are extracted where comparable filing
   balances exist;
3. unresolved counterparties remain explicit placeholders rather than inferred names;
4. Intersect, Hyperion, Saline, River Bend, CoreWeave DCSP, Kenilworth, Cambridge,
   Laidley, and Evest have project-level ownership or dependency records;
5. OpenAI, Anthropic, xAI, Mistral, and Cohere have financing chronologies;
6. AI-dedicated and mixed infrastructure remain labeled separately; and
7. outside debt, leases, guarantees, and project capital are mapped where public.

The 22 critical unknowns remain open because the decisive contracts, private-debt
documents, or unredacted utility schedules are not public in the reviewed record.
Their evidentiary boundaries are nevertheless explicit: undisclosed amounts remain
null; options and announcements remain nonfunded; project cost is not substituted for
debt; maximum guarantees are not expected loss; and aggregate commitments are not
assigned to named counterparties without contract evidence.

## Writing gate

The Chapter 2 research pass clears its writing gate:

- common-period capital and commitment bridges are complete for the anchor entities;
- the largest private financing structures are reconstructed;
- flow/stock overlap is governed by explicit selection rules;
- debt, lease, equity, guarantee, and project-finance sources are mapped;
- unresolved material facts are bounded by null amounts, status labels, overlap rules,
  and named resolution paths; and
- committed-capital, funded-cash, and dependency-graph datasets are frozen with hashes.

Writing must use the frozen graph manifest and may not collapse the two capital bases.
