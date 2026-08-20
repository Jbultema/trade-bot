# Economic Model Governance for Chapters 12–14

This specification governs the economic engine, cash-waterfall, return and utilization
work. It is a calculation contract, not a source of economic inputs. Every input must
trace to a structured claim and carry an evidence status. Missing material inputs make
an output bounded or unresolved; they are not filled with an unlabeled industry
convention.

## 1. Permitted evidence states

Every numerical input and output carries exactly one state:

1. `observed` — directly filed, contracted, paid, recognized or measured;
2. `contract_implied` — arithmetic follows enforceable disclosed terms;
3. `bounded` — lower and upper endpoints are separately evidence-supported;
4. `scenario` — explicit counterfactual input with no implied probability; or
5. `unresolved` — a material input is unbounded and the result cannot be calculated.

An output inherits the weakest material input state. An observed capex amount does not
make an IRR observed. A model built only from Grade-A inputs is still a model.

## 2. Unit-economics identity

For asset or service `a` during period `t`:

`productive_units[a,t] = available_units[a,t] × productive_utilization[a,t]`

`revenue[a,t] = productive_units[a,t] × realized_unit_price[a,t]`

`contribution[a,t] = revenue[a,t] - variable_power[a,t] - variable_network[a,t]
- variable_memory[a,t] - other_variable_cost[a,t]`

`asset_operating_cash_flow[a,t] = contribution[a,t] - fixed_opex[a,t]
- maintenance_capex[a,t] - working_capital_change[a,t]`

The model must not substitute reserved, consumed or billable units for productive
units without naming the substitution and showing the resulting bound.

## 3. Utilization ladder and denominators

The retained ladder is:

`installed → available → delivered → energized → accepted → reserved → consumed →
billable → productive`

Each observation records:

- native quantity unit;
- numerator state;
- denominator state;
- measurement interval;
- average, point-in-time or peak basis;
- owned, leased or service perimeter;
- training, inference or mixed workload;
- customer or portfolio aggregation;
- evidence grade;
- uncertainty and known exclusions.

The following ratios answer different questions:

- `availability = available / installed`;
- `acceptance = accepted / delivered`;
- `reservation = reserved / available`;
- `consumption = consumed / available`;
- `billing = billable / available`;
- `productive utilization = productive / available`.

No ratio is called “GPU utilization” without both states.

## 4. Economic demand

Demand is tested at three independent layers:

- **financial:** a binding payment, floor, prepayment or minimum exists;
- **physical:** the asset has been delivered, energized and accepted; and
- **economic:** productive consumption at realized price supports the required return.

A relationship can be financially supported and physically delivered while remaining
economically uneconomic. Conversely, uncontracted physical capacity may become
economic if repriced and resold. The model retains all three states.

## 5. Capital and useful-life perimeter

Initial invested capital includes only amounts required to place the defined asset or
system in service:

- land and site preparation;
- shell and mechanical/electrical infrastructure;
- power interconnection and dedicated generation;
- compute, memory, networking and storage equipment;
- capitalized interest;
- development fees;
- initial working capital;
- less customer contributions, nonrepayable public support and other explicitly
  identified offsets.

Repayable debt and government loans are financing sources, not reductions of invested
capital. Tax credits reduce after-tax capital only when qualified basis, timing and
cash realization are stated.

Accounting depreciation, tax depreciation, physical life and economic useful life are
four separate observations. Replacement capital follows economic life; accounting
ROIC follows the stated accounting policy.

## 6. Project cash waterfall

The standard unlevered-to-equity waterfall is:

1. customer collections and other operating cash receipts;
2. indirect tax and pass-through amounts;
3. power, network and variable operating cost;
4. fixed operating cost;
5. maintenance and replacement reserve;
6. working-capital requirement;
7. senior operating or project debt interest;
8. mandatory principal amortization;
9. cash sweep and restricted reserves;
10. subordinated debt or preferred return;
11. tax distributions and project tax;
12. residual equity cash flow;
13. terminal or residual value, if separately supported.

Every step records priority, beneficiary, trigger, cap, timing, recourse and source.
Unknown steps remain visible. A project IRR is prohibited when minimum rent, operating
cost, debt service or terminal value is both material and unbounded.

## 7. Return measures

### Project IRR

Calculated from the dated cash flows of the defined project and capital claimant.
Unlevered and levered IRR must not be mixed.

### Accounting ROIC

`NOPAT / average accounting invested capital`

The numerator, tax rate, capital base, goodwill treatment, lease treatment and period
must be stated.

### Incremental ROIC

`incremental attributable NOPAT / average incremental attributable capital`

AI-specific attribution is a model unless directly reported.

### Economic return

The discount rate or value multiple earned by a named capital provider after operating,
financing, tax, replacement and terminal requirements.

### Required-revenue model

Solves for revenue that satisfies:

`present_value(after_tax_free_cash_flow) = initial_private_capital`

This is often more defensible than claiming an observed AI return.

### Implied-return model

Uses bounded revenue, cost, utilization and capital ranges to solve a return interval.
It is unresolved if one material range is open-ended.

### Reverse-DCF model

Solves for the utilization, price, margin, useful life or growth required to justify a
stated asset value or incremental capital stock. The result is a required condition,
not a forecast.

## 8. Hyperscaler attribution models

No single method is publication-sufficient. Each hyperscaler is tested, where possible,
through:

1. accounting attribution;
2. incremental cash-flow attribution;
3. required-revenue analysis;
4. implied-return ranges; and
5. reverse DCF.

The common funding-capacity bridge is:

`operating cash flow - maintenance capital - non-AI growth capital
= indicative internally available AI funding capacity`

Maintenance capital and non-AI growth capital are modeled ranges unless disclosed.
Funding capacity is not evidence that AI investment earns an adequate return.

## 9. First-loss thresholds

For each capital claimant, solve the earliest relevant threshold:

- price decline;
- utilization decline;
- capex overrun;
- power-cost increase;
- delivery delay;
- useful-life shortening;
- refinancing-rate increase;
- terminal-value decline; or
- customer default.

The result names the first-loss bearer, contractual protection, timing and next
claimant. Guarantees and customer contributions move the threshold; they do not erase
the underlying economic loss.

## 10. Required scenarios

At minimum:

- 20% reduction in relevant hyperscaler capex categories;
- 90% inference-price decline;
- one-year delivery delay;
- useful-life compression;
- power and memory cost shock;
- utilization ramp below contract case;
- refinancing spread and advance-rate shock;
- customer default with disclosed support rights; and
- combined shock with no assumed diversification benefit.

Each scenario has a base date, shocked variable, range, lag, affected contract,
loss-bearing entity and recovery mechanism. A scenario does not receive a probability
unless calibrated independently.

## 11. Anti-false-precision rules

- No project IRR with an invented terminal value.
- No AI-specific hyperscaler ROIC from company-wide cloud revenue alone.
- No utilization point estimate from electricity use without the power and hardware
  denominator.
- No revenue inferred from announced capacity without accepted, billable and price
  evidence.
- No public subsidy total that adds grants, loan principal, tax credits, government
  equity and catalyzed private investment.
- No circularity adjustment that assumes an investor-funded customer purchase is
  economically artificial without tracing cash, enforceability and independent value.
- No “systemic” label from network centrality alone.
- No historical-analogue conclusion from a single similarity score.

## 12. Output requirements

Every published model table retains:

- model ID and version;
- as-of date;
- entity, project and exposure IDs;
- claimant and economic beneficiary;
- input value or range;
- unit and currency;
- evidence state and grade;
- source claim and locator;
- transformation;
- sensitivity range;
- output state;
- falsification condition;
- downstream chapter use; and
- unresolved material inputs.

Model code, frozen inputs, outputs and validation hashes are delivered separately from
chapter prose.
