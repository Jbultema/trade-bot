# Chapter 11 research pass: circularity, cross-ownership, and duplicated exposure

Status: staged research packet only  
As of: 2026-07-25  
Writer boundary: this lane does not write canonical ledgers, the database, chapter prose, publication artifacts, or manifests outside this directory.

## 1. Investigative question

This pass asks when apparently circular AI financing is a legally linked cash loop, when it is only reciprocal commercial strategy, and when it is ordinary vertical integration or risk sharing. It therefore separates:

1. funded cash from announced or committed capital;
2. purchase obligations from recognized revenue;
3. supplier investment from customer consideration;
4. guarantees and backstops from funded debt;
5. project debt from tenant, end-user, and utility obligations;
6. present ownership from contingent warrants or future options; and
7. gross reciprocal positions from net accounting presentation.

The pass does not estimate a single “AI circularity total.” The reviewed instruments have different obligors, states, recourse, conditions, and overlap groups. Addition is denied unless legal independence and economic non-overlap are both demonstrated.

## 2. Classification produced by this pass

| Class | Test | Strongest reviewed example | Chapter 11 treatment |
|---|---|---|---|
| Gross reciprocal project finance | Cash or assets move in both directions between the same economic pair and both gross claims remain live | CoreWeave 78 MW DCSP | Documented reciprocal gross finance with partial proceeds recycling; not a closed cash loop |
| Performance-linked financing availability | Supplier delivery creates availability under a customer financing instrument | Amazon–Anthropic | Documented operational-to-financing gate; not evidence of cash recycling |
| Purchase-linked contingent cross-ownership | Customer receives equity rights only when purchases or performance milestones occur | AMD–OpenAI | Conditional consideration; not present ownership |
| Reciprocal capital and demand | Investor/supplier and customer obligations coexist without public tracing of the same dollars | Microsoft–OpenAI; Amazon–OpenAI; Microsoft/NVIDIA–Anthropic | Role convergence; cash recycling unproven |
| Conditional residual-capacity support | Supplier or investor buys only qualifying capacity left unsold by other customers | NVIDIA–CoreWeave | Utilization backstop; not a guarantee of all demand |
| Customer-funded or customer-supplied hardware | Customer prepays or supplies GPUs used to deliver contracted service | Oracle AI customer cohort | Mixed cash and in-kind funding; customer allocation unresolved |
| Layered risk-transfer chain | Developer; asset owner; tenant; guarantor; end user; utility; and lender have distinct legal roles | River Bend; Saline | Dependency chain; not a closed loop |
| Progressive announced coupling | Capital intent is contingent on future deployment and definitive documentation remains incomplete | NVIDIA–OpenAI 10 GW LOI | Announced ceiling only |
| Strategic vertical integration | Buyer invests in a supplier while reserving capacity or co-developing products | NVIDIA–Coherent; NVIDIA–Marvell | Ordinary strategic finance unless cash recycling is separately shown |

## 3. High-confidence findings

### 3.1 CoreWeave’s 78 MW provider structure is the strongest documented reciprocal gross-finance case

CoreWeave’s March 2026 10-Q describes one data-center service provider that agreed to design, acquire, build, and manage up to 78 MW. Four unlike amount states remain separate: a $305 million seven-year 13% senior-secured delayed-draw loan commitment; a $303 million recognized gross note receivable at March 31, 2026; a $116 million purchase by the provider of existing CoreWeave critical infrastructure; and a $232 million combined financing-obligation and finance-lease balance. The provider used borrowing to fund only part of the asset purchase. The filing does not provide a dollar-for-dollar facility draw and use-of-proceeds schedule. CoreWeave retained control and recorded a fourteen-year financing obligation, while a separate fourteen-year finance lease also runs from the provider back to CoreWeave.

This is not merely overlapping roles. It contains demonstrable opposing gross asset, loan, financing-obligation, and lease claims between the same two economic nodes, with partial proceeds recycling documented for the asset purchase. It is strongest relative to the reviewed cases for that reason—not because a fully quantified closed cash loop is proven. Net presentation does not extinguish the counterparty, construction, collateral, capacity, or tenant exposures. The facility draw/use schedule, unidentified provider, asset-title chain, collateral realization path, and replacement-cost perimeter remain material unknowns and explicit falsification/resolution targets.

Evidence: L3-SRC-0012 / canonical SRC-0015; CoreWeave 10-Q; Note 10 “DCSP Financing Arrangements.”

### 3.2 Amazon–Anthropic is the strongest performance-gated financing-availability case

Amazon’s quarter ended March 31, 2026 and the transaction/disclosure state dated April 20, 2026 are recorded separately. The post-quarter disclosure leaves three exact and nonadditive capital states: $13 billion of combined funded notes and preferred investment with the instrument allocation unresolved; a separate future facility with a $20 billion ceiling and zero availability at inception; and a separate Amazon option with a $5 billion ceiling that reduces rather than adds to facility availability. Availability is earned as Amazon achieves specified compute-capacity delivery milestones; draws cause Anthropic to issue convertible notes or common stock for cash.

Anthropic separately announced a ten-year AWS commitment above $100 billion and up to 5 GW, while retaining AWS as its primary training and cloud provider. Those amounts are not cash received by Anthropic. The documentary fact is narrower: Amazon’s own supplier performance gates the availability of its future investment capital. This is performance-linked financing availability, not proof that any AWS payment returns as Amazon investment.

Evidence: L3-SRC-0005 / canonical SRC-0013; Amazon 10-Q; Note 2 subsequent-events Anthropic disclosure. L3-SRC-0006 / canonical SRC-0056; paragraphs describing 5 GW, greater-than-$100 billion AWS commitment, and investment amounts.

### 3.3 OpenAI relationships show several different mechanisms, not one loop

- Microsoft: $13 billion had been committed and $11.8 billion funded at March 31, 2026. A separate $250 billion incremental Azure purchase exists. The April 2026 amendment retained Microsoft as primary cloud, allowed OpenAI products across clouds, made the IP license nonexclusive, ended Microsoft-to-OpenAI revenue sharing, and retained capped OpenAI-to-Microsoft revenue sharing through 2030. The public record does not trace OpenAI’s Azure or revenue-share payments to Microsoft’s original investment dollars.
- Amazon: Amazon funded a $15 billion OpenAI Series C purchase and committed another $35 billion under event and outside-date conditions. OpenAI also has an existing $38 billion AWS commitment plus a $100 billion expansion. These are distinct equity and service-contract states. The public record does not provide a customer-by-customer cash provenance bridge.
- NVIDIA: the September 2025 document is a letter of intent. It states an intent to invest progressively up to $100 billion as at least 10 GW is deployed; definitive terms were still being finalized. It is not a funded commitment and must not be added to a later OpenAI financing allocation without instrument-level reconciliation.
- AMD: OpenAI received a warrant for up to 160 million AMD shares at a $0.01 exercise price. Vesting is tied to GPU purchases, delivery through 6 GW, AMD share-price thresholds, and technical/commercial conditions. This is purchase-linked contingent consideration; it was not present funded equity or present voting ownership at March 28, 2026.
- Oracle: Oracle disclosed that $75 billion of large AI contract value was supported either by customer prepayments used to buy GPUs or by customers buying and supplying GPUs. The two modes are economically distinct and the published aggregate is not allocated to OpenAI. Oracle’s named-customer financing plan and the Saline project do not make the $75 billion an OpenAI-specific cash flow.
- CoreWeave: the filed MSA exposes transition, mitigation, resale/reconfiguration, received-service payment, prepayment setoff, and force-majeure refund mechanics. The public order-value ceiling is not proof of cash funded, capacity accepted, or revenue recognized.

### 3.4 NVIDIA–CoreWeave is correlated multi-role exposure, but not a proven closed cash loop

NVIDIA invested $2 billion in CoreWeave equity; supplied the GPU platform used across CoreWeave’s then-deployed fleet because customer contracts specified NVIDIA systems; and agreed to buy up to $6.3 billion of qualifying residual unsold capacity through 2032. CoreWeave also finances GPU-rich infrastructure with customer-linked delayed-draw debt; aggregate DDTLs outstanding were $11.8 billion at March 31, 2026, and the $8.5 billion DDTL 4.0 was only partly drawn.

The residual order applies only when ordinary customer use does not fill qualifying capacity and remains subject to delivery, availability, breach, and insolvency conditions. It is therefore a utilization backstop, not a guarantee of all CoreWeave demand or debt service. The equity, hardware purchases, customer contracts, residual order, and project debt are correlated components and are nonadditive by default.

Evidence: L3-SRC-0012 through L3-SRC-0015; Note 2; equity-financing discussion; supplier-risk disclosure; filed residual-capacity 8-K; filed OpenAI MSA.

### 3.5 Developer–hyperscaler–credit–utility chains partition risk rather than close a loop

River Bend separates the issuer/landlord (Hut 8 DC), direct tenant (Fluidstack), specified tenant-payment backstop (Google), downstream model-company demand (Anthropic), and project noteholders. The $3.25 billion project notes are funded debt; the $7 billion base-lease value is future contracted revenue; the optional $17.7 billion including renewals is not additive to the base value. Anthropic is not publicly identified as project-debt obligor or direct tenant.

Saline separates Related/Blackstone project sponsorship, PIMCO-anchored announced debt, Oracle infrastructure tenancy, Green Chile Ventures as legal DTE utility customer/operator, OpenAI end use, and DTE’s separately owned or procured storage. Green Chile’s primary supply agreement has a maximum 1,383 MW ramp, 80% minimum billing demand, parent support, and potential letter of credit. The public documents do not make OpenAI the utility obligor, campus borrower, or storage owner.

These chains can transmit impairment, but circularity is not the right default label. Their core feature is layered credit substitution and risk allocation across legally distinct nodes.

## 4. Alternative explanations explicitly tested

1. **Ordinary vertical integration.** A dominant platform vendor may invest in an adjacent supplier to secure capacity or product compatibility. NVIDIA–Coherent and NVIDIA–Marvell fit this explanation unless evidence shows that the supplier’s sales finance the investor’s investment or that purchase consideration and equity are legally netted.
2. **Risk sharing.** A guarantee, capacity backstop, prepayment, or take-or-pay lease can make a new build financeable without recycling the same cash. River Bend and Saline principally fit layered risk sharing.
3. **Commercial incentive.** A deeply in-the-money warrant may reduce the effective economics of a purchase while remaining contingent and noncash. AMD–OpenAI fits this category.
4. **Customer financing.** A supplier may fund its customer to expand adoption. Amazon–Anthropic is stronger than this generic label because supplier delivery explicitly creates facility availability.
5. **Accounting presentation.** Netting can obscure opposing gross positions even when no additional cash moves at the reporting date. CoreWeave’s DCSP is the strongest reviewed reciprocal gross-finance example, with only partial proceeds recycling documented.

## 5. System-level interpretation boundary

The proposed Grade-C case-indexed evidence matrix covers all fourteen reviewed cases. It finds that the available public evidence does not currently jointly establish the editorial test for a supported “systemically consequential” designation in any case. This is an evidence-limit conclusion, not evidence that systemic consequence is absent. It remains noncanonical pending primary review. The missing elements are:

- impairment transmission across at least three independently financed parts of the stack;
- inability to replace the impaired node within the relevant stress horizon; and
- a calibrated loss or cash-flow shortfall large enough to change capacity, credit allocation, or loss allocation beyond the directly exposed parties.

The CoreWeave–NVIDIA–customer–lender configuration is the strongest candidate for a future multi-node stress test. The River Bend and Saline chains are strong project-level transmission cases. Amazon–Anthropic is a strong availability-coupling case. A second proposed Grade-C matrix shows why no cross-case additive total is currently authorized: the cases contain overlapping gross exposures and incompatible amount states. That boundary does not net away or ignore gross claims. Both cross-case conclusions remain proposed and noncanonical, while the classification taxonomy is methodology rather than a factual claim.

## 6. Amount-state and overlap controls

- Announced project cost, capacity, or financing intent is not contracted capital. Saline is recorded as an exact $16 billion announced project-cost perimeter with funded amount unresolved, and remains separate from DTE storage and utility obligations.
- Contracted service value is not funded cash, drawn debt, recognized revenue, or productive utilization.
- Equity funded, equity committed, warrant not vested, and future option are separate states.
- Guarantees, residual orders, and backstops are contingent maximums; they are not principal funded.
- Customer-supplied hardware is in-kind and cannot be added to customer cash prepayments without a disclosed split.
- Base lease value and optional renewal value overlap.
- Financing envelopes and vendor backstops are nonadditive unless documents allocate instrument shares.
- Gross receivables and gross reciprocal obligations must remain visible even when accounting permits net presentation.
- Legal obligor, economic beneficiary, accounting holder risk, and economic first-loss order are separate fields. Across all 30 proposed exposures, 25 economic first-loss entries remain `not_determined`. Five funded or recognized assets identify only the named holder's accounting security-value or receivable-impairment risk and explicitly do not claim a default waterfall. No sponsor, customer, guarantor, borrower, shareholder, or mixed package is assigned economic first loss without direct security, intercreditor, recovery, and claimant-order evidence.

## 7. Downstream work

Chapter 13 should model cash-flow pressure or first loss only after obligor, security, claimant-order, payment-waterfall, recovery, cancellation, and utilization states are resolved. Chapter 16 should treat role convergence as a conditional transmission map, not as proof of a closed loop or loss allocation. Chapter 17 should distinguish disclosure architecture from substantive market concentration and should identify which contract schedules, intercreditor agreements, UCC records, utility dockets, and order forms would resolve the highest-materiality unknowns.

Detailed proposed rows, falsifiers, unknowns, evidence links, and handoffs accompany this memo. They are candidates for primary-agent review and deduplication, not canonical facts.
