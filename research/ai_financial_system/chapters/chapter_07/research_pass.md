# Chapter 7 Research Pass

## Provisional title

**Hyperscaler Balance Sheets — Capex, Leases, Purchase Commitments, and Customer
Concentration**

## Research question

How much AI infrastructure exposure is actually recognized, contracted, financed,
leased, guaranteed or left outside the balance sheets of Microsoft, Amazon, Alphabet,
Meta and Oracle—and what cash-flow, maturity, customer and counterparty structures
determine who absorbs underutilization or delivery failure?

## Pass 1 scope

Reconstruct each hyperscaler's latest available public-company perimeter using:

- balance sheet property and equipment, finance leases, operating leases and debt;
- cash-flow purchases of property and equipment, acquisitions and supplier financing;
- construction-in-progress, useful lives, depreciation and capitalized interest;
- purchase commitments, cloud-capacity obligations, take-or-pay agreements and RPO;
- customer prepayments, deferred revenue, significant financing components and
  customer-supplied equipment;
- VIEs, project guarantees, off-balance-sheet power and data-center arrangements;
- capex guidance reconciled to filed cash-flow and balance-sheet movements;
- disclosed customer or supplier concentration and unnamed large-AI-customer exposure;
- maturity ladders and the lag from commitment to delivery to revenue.

## Required comparison bridges

1. **Capex bridge:** cash capex → PP&E additions → construction in progress →
   depreciation → disposals / acquisitions → ending PP&E.
2. **Obligation bridge:** recognized debt and leases → purchase commitments →
   guarantees → customer-funded capacity → non-additive maximum exposure.
3. **Revenue-funding bridge:** RPO and deferred revenue → prepayment →
   customer-supplied hardware → recognized revenue → cash collection.
4. **AI-attribution bridge:** disclosed AI-specific amount → data-center or cloud
   proxy → company-wide residual → explicitly unallocable amount.
5. **Risk bridge:** customer delay or underuse → contract remedy → asset
   reconfiguration → impairment / stranded cost → guarantor, investor or ratepayer.

## Initial entity set

- Microsoft Corporation
- Amazon.com Inc.
- Alphabet Inc.
- Meta Platforms Inc.
- Oracle Corporation

Other entities enter only where needed to identify financing, construction, energy,
leasing, equipment or customer concentration around those balance sheets.

## Non-negotiable measurement rules

- Capex guidance is not filed cash capex.
- PP&E additions are not necessarily cash purchases in the same period.
- Lease liabilities are not purchase commitments.
- RPO is not funded customer cash.
- Customer prepayments are not revenue until recognition criteria are met.
- Customer-supplied GPUs are not hyperscaler-owned capex.
- Maximum guarantees are not expected loss.
- Company-wide data-center assets are not automatically AI-specific.
- Fiscal years and trailing periods must be calendar-aligned before comparison.

## Writing gate

Chapter 7 cannot enter writing until:

- all five company cases use the latest available primary filings as of the freeze date;
- each case has a cash-capex / PP&E / depreciation bridge or a documented reason it
  cannot be reconstructed;
- recognized leases and debt are separated from commitments and guarantees;
- Oracle's customer-prepayment and customer-supplied-GPU model is reconciled separately;
- AI attribution and unallocable residuals are explicit;
- at least one cross-company maturity and risk-transmission dataset is frozen;
- all material statements are atomic claims linked to source records;
- contradictions, superseded guidance and unresolved questions are preserved.

**Gate result — passed 25 July 2026.** The writing pass uses the frozen database
snapshot with SHA-256
`4dcef741342d3996bdb5d07cee54ae0f2aa50f9f2a6e4c0ae5f301b16beba314`.

## Pass 1A — Microsoft: the lease pipeline is larger than the recognized lease balance

### Gross asset growth has three funding clocks

Microsoft's property and equipment at cost rose from $298.619 billion at June 30, 2025
to $394.951 billion at March 31, 2026, an increase of $96.332 billion. Servers, network
equipment and software increased by $58.047 billion; buildings and improvements
increased by $34.339 billion. Net PP&E reached $283.228 billion after $111.723 billion
of accumulated depreciation.

The nine-month cash-flow statement shows $80.146 billion of cash additions. The lease
note separately shows $19.486 billion of finance-lease right-of-use assets obtained.
Purchases of property and equipment remaining in accounts payable increased to
$22.6 billion from $6.9 billion. These values do not sum cleanly because one is a period
cash flow, one is noncash lease commencement, and one is an ending payable. Together,
they show why cash capex alone understates asset formation and why adding every number
would overstate it.

### Uncommenced leases are a second infrastructure pipeline

Recognized lease liabilities at March 31 were $22.238 billion operating and $62.932
billion finance, both on a present-value basis. Microsoft separately disclosed
$196.6 billion of additional leases, primarily for data centers, that had not commenced.
Those contracts are expected to commence between fiscal 2026 and fiscal 2031 with terms
of one to twenty-one years.

The $196.6 billion does not belong on the current balance sheet because the right to use
the assets has not commenced. It is also not directly comparable to the present-value
lease liabilities without a payment schedule and discounting. Economically, however,
it is a signed future capacity pipeline that can become recognized assets, liabilities,
depreciation, interest and cash payments as facilities are delivered.

### Supplier support appears as financial assets, not data-center capex

Microsoft reported $17.8 billion of other receivables from activities facilitating the
purchase of server components, up from $8.2 billion, plus $11.5 billion of restricted
investments under a supplier agreement. The receivables and restrictions sit in
financial-asset accounts rather than PP&E.

This is a separate capital channel:

`Microsoft liquidity → supplier-linked receivable or restricted investment → component
availability → Microsoft or partner server acquisition → data-center capacity`

The counterparties, security, recourse and overlap between the two assets are not
allocated. They cannot be called vendor loans without the agreements, but they
demonstrate that capital committed to infrastructure can remain outside the capex line.

### RPO and unearned revenue are not equivalent funding

Commercial remaining performance obligations reached $627 billion, while total
unearned revenue was $53.677 billion. Microsoft expects to recognize approximately 25%
of commercial RPO over the following twelve months. RPO includes future amounts not yet
invoiced; unearned revenue is a recognized contract liability for amounts billed or
paid before performance.

The difference is economically important. Neither amount identifies cash restricted to
data-center construction, customer concentration, contract cancellation, or the margin
earned when service is delivered.

## Pass 1B — Amazon: most current asset formation is in AWS, but not all AWS is AI

### Cash purchases and net additions diverged by more than $10 billion in one quarter

Amazon reported $54.757 billion of consolidated net PP&E additions in first-quarter
2026 and $44.203 billion of cash purchases. It separately reported $1.565 billion of
finance-lease additions and a $9.920 billion increase in PP&E acquired but not yet paid.
Those noncash items explain much of the difference but are not a forced complete
reconciliation because sales, incentives, remeasurements and other activity can also
matter.

The same distinction appears in trailing cash flow. Operating cash flow was $148.531
billion for the twelve months ended March 31, while cash property purchases net of sales
and incentives were $147.299 billion. Amazon's resulting free-cash-flow measure was
$1.232 billion. Amazon expressly says that measure omits finance-lease principal, debt
reduction, acquisitions and other components.

### AWS received three quarters of current PP&E additions

AWS held $223.056 billion of net PP&E at March 31, approximately 56.1% of the
consolidated $397.458 billion. It received $41.516 billion of first-quarter net
additions, approximately 75.8% of the consolidated total, and recorded $7.277 billion
of PP&E depreciation and amortization, approximately 56.7% of the consolidated amount.

Amazon allocates shared technology infrastructure among segments based on usage. The
AWS values therefore provide a cloud-infrastructure perimeter, not an AI-only perimeter.
They include external AWS services and infrastructure used by Amazon's other businesses.

### The commitment table mixes six different balance-sheet states

Amazon reported $104.942 billion of recognized present-value lease liabilities and
$106.347 billion of leases not yet commenced. Its $569.280 billion total contractual
commitment table additionally includes debt principal and interest, finance and
operating lease gross payments, financing obligations, $103.768 billion of unconditional
purchase obligations and $18.813 billion of other commitments.

The aggregate is useful as a maturity envelope, not as incremental off-balance-sheet
exposure. Several component rows are already recognized liabilities, and debt interest
is not balance-sheet principal. Unconditional purchase obligations also combine energy,
property and equipment, media and software.

### Customer backlog is much larger than current free cash flow

Amazon reported approximately $364 billion of primarily AWS performance obligations for
contracts longer than one year, with a 5.5-year weighted remaining life. The same note
described OpenAI's existing $38 billion commitment plus a $100 billion expansion over
eight years. It does not allocate RPO by customer or prove that the full $138 billion is
a noncancellable minimum included in RPO.

The apparent hedge is therefore incomplete:

`large customer contract → future AWS revenue → recovery of data-center capital`

The public record omits cash timing, capacity acceptance, minimum volume, termination,
service credits and customer concentration. RPO demonstrates contracted demand, not
perfect asset recovery.

### Technology and infrastructure expense is broader than AI

First-quarter technology and infrastructure expense was $29.567 billion. It includes
research and development plus server, network and data-center depreciation, rent,
utilities and related expense. The majority of infrastructure cost is allocated to AWS,
but the category also supports Amazon's internal systems, devices, satellites,
autonomous vehicles and other initiatives. Treating it as “AI opex” would over-attribute
the filed amount.

## Pass 1C — Alphabet reconstruction

### The largest visible construction queue is not yet depreciating

Alphabet's first-quarter cash purchases of property and equipment were $35.674 billion,
but the more revealing stock measure is its $108.597 billion of assets not yet in
service. That population rose $30.005 billion in one quarter. It sits outside the
in-service asset base and therefore generally has not entered its ordinary productive
and depreciation phase.

This does not mean Alphabet “spent $108.6 billion in the quarter.” It means the balance
sheet contains a large queue of acquired and constructed capacity whose service dates,
workload assignments and customer support are not disclosed. The queue is economically
important because the cash and construction risk can precede the revenue and depreciation
profile by many months.

Alphabet also reported $24.131 billion of property and equipment in accounts payable and
accrued expenses. Cash capex therefore captures neither the complete timing of asset
formation nor the complete supplier-funding channel.

### Lease commitments and project backstops are separate layers

At March 31, Alphabet recognized $16.161 billion of operating lease liabilities and
$2.214 billion of finance lease liabilities. Another $75.6 billion of primarily
data-center leases had not commenced.

Separate disclosures describe $332.4 billion of purchase commitments and contractual
obligations, including technical infrastructure, inventory, content and energy
take-or-pay arrangements. Alphabet also reported maximum exposures of $9.0 billion under
guarantees and $28.4 billion under credit derivatives associated with infrastructure
arrangements. Some remedies can require Alphabet to assume or sublease an underlying
lease or make a termination payment.

These are not three additive loss numbers. The guarantee and derivative exposures may
support projects that also appear in lease or infrastructure commitments. Their
importance is structural: third-party project and lease financing can migrate back
toward Alphabet if a supported counterparty fails.

### Backlog is concentrated in Cloud but is not cash collateral

Alphabet reported $467.6 billion of total RPO, of which $462.3 billion belonged to Google
Cloud. More than half of total RPO was expected to be recognized within twenty-four
months. This is powerful evidence of contracted demand, but public disclosures do not
map that demand to the $108.6 billion construction queue, the $75.6 billion future lease
population or the infrastructure backstops.

## Pass 1D — Meta reconstruction

### Meta combines a large owned pipeline with even larger future access commitments

Meta reported $194.776 billion of net property and equipment, including $61.017 billion
of construction in progress. First-quarter cash property purchases were $18.997 billion,
while $16.040 billion of property and equipment remained in accounts payable and accrued
expenses.

The fixed-asset stock includes $105.987 billion of servers and network assets at cost,
$58.836 billion of buildings and $8.842 billion of finance-lease ROU assets. These
categories are company-wide and support both AI and core recommendation, ranking,
advertising and other services.

Meta's future-use perimeter is larger than its current recognized asset stock:

- $182.88 billion of uncommenced data-center, colocation and network leases;
- $237.67 billion of noncancelable contractual commitments, mostly third-party cloud
  capacity, servers, networks, data centers and Reality Labs hardware;
- up to $14.72 billion of contingent cloud purchases that can be reduced by provider
  resale; and
- another $24 billion of multi-year infrastructure commitments signed after quarter end.

The filing does not establish whether the first two populations overlap. They therefore
remain separately classified and nonadditive.

### Owned construction can migrate into third-party development

Meta classified $1.48 billion of primarily construction in progress and land as held for
sale because it planned to contribute those assets to a third party for joint data-center
development. This is a small balance relative to Meta's total build, but a significant
control observation: an owned construction asset can leave the consolidated PPE path
and reappear economically through a joint-development, lease, service or guarantee
relationship.

The counterparty, consideration, retained rights and replacement obligations are not
publicly disclosed. A conventional capex chart cannot show whether this migration reduces
risk or merely changes its accounting location.

### Guidance does not provide an AI-only denominator

Meta guided to $125 billion to $145 billion of 2026 capital expenditures “to support both
our AI efforts and our core business.” That wording blocks an AI-only attribution.
Company-defined first-quarter capex was $19.84 billion, versus $18.997 billion of cash
property purchases, because the company measure also includes finance-lease principal.
Neither amount should be presented as pure AI investment.

## Pass 1E — Oracle reconstruction

### Oracle is the clearest case of buildout exceeding internally generated cash

Oracle spent $55.663 billion on capital expenditures in fiscal 2026 against $31.977
billion of operating cash flow. Its supplemental bridge reduced the company cash outlay
to $47.726 billion after $3.345 billion of short-term manufacturer financing and $4.592
billion of customer prepayments with a significant financing component.

The bridge is informative precisely because the categories are economically different:

`asset acquisition → supplier financing + customer financing + Oracle cash`

The customer prepayment is cash but also creates deferred revenue and a performance
obligation. The manufacturer financing is a funding channel, not a reduction in the
asset's economic cost.

Oracle finished the year with $99.957 billion of net property and equipment, including
$39.973 billion of construction in progress and $7.464 billion of finance-lease ROU
assets. It also raised $43 billion of debt and $5 billion of equity during fiscal 2026.

### The off-balance-sheet lease queue is larger than Oracle's recognized asset base

Oracle recognized $30.190 billion of operating lease liabilities and $7.701 billion of
finance lease liabilities. It separately disclosed $260 billion of substantially
data-center leases not yet commenced. These leases generally begin from fiscal 2027
through fiscal 2029 and run fifteen to nineteen years.

One future lease includes a guarantee of up to $3.3 billion of the lessor's borrowing.
That guarantee is embedded in the future lease population; adding it would double count
part of the same project structure.

Oracle also carried $13.309 billion of unconditional obligations primarily for
data-center power and signed another $19 billion of five-year cloud-infrastructure asset
purchase commitments after year end.

### Customer-funded GPUs reduce capital need but do not eliminate duration mismatch

Oracle reported $638 billion of RPO, up from $138 billion one year earlier. Management
said the prepaid and customer-supplied hardware portions of large AI contracts totaled
$75 billion. This is not $75 billion of unrestricted cash: the disclosure combines cash
prepayments with GPUs purchased and supplied by customers, without disclosing the split
or equipment title.

Oracle's own risk factor states the central mismatch. Long-term data-center lease terms,
renewal options and pricing adjustments may not align with customer-contract durations
and pricing. If customers do not renew, Oracle may be unable to re-lease, repurpose or
assign capacity on acceptable terms.

The reconstructed chain is therefore:

`large AI contract → customer cash or equipment + Oracle external financing → owned and
leased capacity → service delivery → customer renewal or residual-capacity risk`

Customer funding reduces the first-stage cash burden. It does not erase the final-stage
asset, lease, power or utilization exposure.

## Pass 1F — Cross-company shadow-capacity bridge

The five reviewed companies disclosed approximately $821.427 billion of leases not yet
commenced:

| Company | Uncommenced leases ($bn) | Stated perimeter |
|---|---:|---|
| Microsoft | 196.600 | Primarily data centers |
| Amazon | 106.347 | Fulfillment, data centers and other mixed use |
| Alphabet | 75.600 | Primarily data centers |
| Meta | 182.880 | Data centers, colocation and network |
| Oracle | 260.000 | Substantially data centers |
| **Arithmetic total** | **821.427** | **Non-harmonized maturity envelope** |

This is a discovery about the scale of the shadow capacity pipeline, not a claim that the
companies have $821.4 billion of comparable AI debt. The values can use different payment
conventions and commencement assumptions; Amazon includes material non-data-center
assets; and none should be added to recognized lease liabilities without discounting and
overlap reconciliation.

The consistent economic feature is that major capital commitments remain outside the
recognized lease liability until the underlying use right commences. The capacity
pipeline can therefore be much larger than the balance-sheet lease liability visible at
the reporting date.

## Pass 1G — Recognized corporate debt

The companies also carried approximately $429.956 billion of recognized corporate debt
at their latest filing dates:

| Company | Recognized carrying amount ($bn) | Attribution |
|---|---:|---|
| Oracle | 129.541 | OCI expansion and general corporate; not allocated by project |
| Amazon | 121.906 | General corporate; latest-quarter infrastructure attribution unavailable |
| Alphabet | 79.499 | General corporate; latest-quarter infrastructure attribution unavailable |
| Meta | 58.748 | General corporate; infrastructure attribution unavailable |
| Microsoft | 40.262 | General corporate; infrastructure attribution unavailable |
| **Arithmetic total** | **429.956** | **Not an AI-debt estimate** |

This layer is deliberately separated from leases, financing obligations, project debt
and principal-plus-interest commitment schedules. Amazon's $203.538 billion debt
commitment schedule, for example, includes future interest and cannot replace its
$121.906 billion recognized carrying amount.

The debt changes nevertheless matter. Amazon issued $37 billion and approximately
$16.8 billion of dollar and euro notes in first-quarter 2026. Alphabet issued dollar,
sterling and Swiss-franc notes with approximately $31.143 billion of translated face
value. Oracle raised $43 billion of debt during fiscal 2026 and explicitly connected its
capital program to AI cloud infrastructure.

The evidence does not support treating every dollar as AI financing. It does support a
more limited conclusion: the infrastructure cycle is increasingly coincident with large
corporate debt issuance, and only Oracle provides a broad program-level link between the
financing and OCI capacity. Project-level sources and uses remain unknown.

## Pass 1H — Filing-currency audit

The latest-periodic-filing gate was rechecked on July 25, 2026:

| Company | Filing used | Period end | Filed | Freeze status |
|---|---|---|---|---|
| Microsoft | Form 10-Q | March 31, 2026 | April 30, 2026 | Latest available; fiscal 2026 Form 10-K not yet filed |
| Amazon | Form 10-Q | March 31, 2026 | April 30, 2026 | Latest available |
| Alphabet | Form 10-Q | March 31, 2026 | April 30, 2026 | Latest available |
| Meta | Form 10-Q | March 31, 2026 | April 30, 2026 | Latest available |
| Oracle | Form 10-K | May 31, 2026 | June 22, 2026 | Latest available |

The frozen comparison is therefore filing-current but not calendar-period-identical.
Microsoft, Amazon, Alphabet and Meta are measured at March 31 while Oracle is measured
at May 31. Cash-flow periods range from one quarter to nine months, a trailing year and a
fiscal year. The chapter will not annualize them because annualization would create
false comparability during a rapidly accelerating build cycle.
