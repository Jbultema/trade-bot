# Chapter 12 research pass — The Economic Engine

Status date: 2026-07-25. This is a staging-only evidence-to-model package, not final chapter prose or a published model.

## Observed facts admitted as model inputs

- River Bend has observed project debt principal ($3.25bn), coupon (6.192%), maturity/amortization terms, a 245-MW base lease, and a $7bn stated 15-year base lease value. Its stated $6.9bn cumulative NOI / $454m average annual NOI is management forecast, not observed cash flow (CLM-040001–004).
- The Saline contract provides a 1,383-MW maximum load ramp, 80% minimum billing demand, term through February 2045, and a formulaic storage revenue requirement, but no public tariff-price, project-cost, or payment schedule (CLM-040008–012; CLM-050017–028).
- OpenAI's $2bn monthly revenue statement, Anthropic/xAI $1.25bn monthly compute fee, and the combined SpaceX AI segment's revenue, operating loss, capex, depreciation and infrastructure-expense changes are observed only at their stated contract or reporting perimeters (CLM-060002, 060013, 060015–019).
- Hyperscaler cash capex, OCF, asset-stock, lease, RPO, prepayment and commitment disclosures are company-wide or segment-wide mixtures; none provides an AI asset cohort's revenue, utilization, price, variable cost, useful life, maintenance capex, or claimant cash flow (CLM-070001–027).

## Contract-implied diagnostics, not project returns

- The Saline **maximum-ramp arithmetic minimum-billing capacity** is 1,106.4 MW (`0.80 × 1,383 MW`, Grade C), only when the contracted ramp reaches 1,383 MW and before tariff charges/contractual adjustments. It is not current load, dollar revenue, energy delivered, utilization, or productive compute demand; ramp timing and actual energized load remain unresolved.
- River Bend's stated $7bn base lease value divided by its stated 15-year base term is an **approximately $466.7m/year nominal full-term aggregate average** only. It is not a period payment, annual contracted rent, recognized revenue, collections, NOI, debt-service coverage, or a dated cash-flow series; the disclosed 3% escalator makes equal-period cash treatment unsafe.
- River Bend's **approximately $201.24m/year annualized pre-amortization simple nominal coupon expense** (`$3.25bn × 6.192%`) assumes full principal is outstanding for a full coupon year. Interest starts in November 2026 and principal amortization in May 2028; it is not April-2026 cash paid, actual debt service, project financing cost, DSCR, or project cash flow.

## Calculability test

| Candidate output | State | Result |
|---|---|---|
| Saline maximum-ramp arithmetic minimum-billing capacity | contract_implied / Grade C | Calculable only conditional on the 1,383-MW maximum contracted ramp: 1,106.4 MW. Not a current-capacity or monetary unit-economic output. |
| River Bend full-term lease aggregate average and annualized coupon run rate | contract_implied / Grade C | Calculable only as conditionally labelled, rounded diagnostic arithmetic. Not period revenue, NOI, DSCR, cash flow, break-even, or IRR. |
| Asset/service revenue per productive unit | unresolved | No matched productive denominator and realized price for any case. |
| Contribution margin / operating cash flow | unresolved | No matched variable power/network/memory cost, fixed opex, working-capital, or maintenance capex perimeter. |
| Project IRR, equity IRR, first-loss utilization, required revenue | unresolved | River Bend lacks operating costs, lease cash schedule, terminal value and complete waterfall; all other observed cases have at least one unbounded material input. |
| AI-specific hyperscaler ROIC | unresolved | No asset/use/revenue/cost attribution; company-wide cloud or capex figures cannot substitute. |

No project or asset currently satisfies the Chapter 12 writing gate for a fully traceable bounded unit-economic model. The useful current output is therefore a disciplined input/readiness map plus limited contract-implied diagnostics.

## Model perimeter and anti-double-counting rules

1. A physical MW/GPU quantity is never placed in a monetary exposure or revenue equation until it is joined to a dated price, ownership/use perimeter and productive state.
2. Capital supplied, debt drawn, lease liability, customer RPO, prepayment, power commitment and project cost are separate observations. Components are nonadditive by default.
3. The SpaceX AI segment combines X and xAI; it cannot establish stand-alone xAI economics. Hyperscaler company and cloud segments include AI and non-AI workloads.
4. River Bend debt, Fluidstack lease, Google support, and Anthropic demand are different claimants/roles. Saline campus and DTE storage are distinct capital pools.

## Falsifiers and evidence queue

The material result would change if an asset-level contract/telemetry package supplies matched productive units, realized price, cost, asset life, cash waterfall and capital claimant. The highest-priority documents are River Bend lease/operating/guarantee schedules; Saline tariff/storage cost and load records; laboratory cloud order forms and audited accounts; hyperscaler asset-cohort ledgers; and accelerator/rack BOM, useful-life and resale records.

## Confidence and economic importance

Baseline established; earlier confidence history not retained. Observed inputs retain their canonical evidence grade. Derived diagnostics are contract-implied and do not inherit Grade A. The inability to compute a unit return is high/structural because it blocks the capital-recovery question; it is not evidence that returns are negative.

## Handoff

CH13 receives only named debt/cash-priority inputs, diagnostic-only arithmetic, and blocked return fields. CH14 receives the full capacity-state schema with mostly unknown measured states and blocked price/productive-utilization fields. CH17 receives selected control/right and cost-allocation triggers without loss-severity estimates.
