# Trade Bot V2.2 AI Review Whitepaper

Status: independent-review packet. Evidence cut: 2026-07-21. This document is
written for a technically capable language model asked to audit the system,
identify circularity or leakage, challenge its empirical conclusions, and
propose bounded improvements. It is not a marketing document and it is not
investment advice.

## 0. Reviewer Contract

The reviewer should separate four questions that are easy to conflate:

1. Is the implementation internally correct and causally faithful?
2. Is a model output historically discriminative?
3. Does acting on that output improve portfolio utility after costs?
4. Is the evidence strong enough for prospective or live authority?

A positive answer to an earlier question does not imply a positive answer to a
later one. A correctly implemented indicator may have no predictive value. A
predictive warning may be too costly to trade continuously. A useful backtest
may still be fragile to execution timing, current-universe survivorship, or
researcher selection. Trade Bot deliberately uses different labels for these
states.

When reviewing, use the following evidence vocabulary:

- `descriptive`: explains a current or historical state; no causal authority.
- `retrospective`: evaluated on known history; useful for rejection or
  hypothesis generation, not prospective proof.
- `walk_forward`: selection or calibration at each origin uses only earlier
  observations, but the modern feature and candidate universe may still be
  known today.
- `prospective`: frozen before the observations being evaluated.
- `allocation_authoritative`: permitted by configuration to change target
  weights.
- `informational_only` or `research_only`: visible but prohibited from sizing.
- `promotion_ready`: not established for the leading i111 challenger.

The requested review should be adversarial. Prefer a precise negative result to
a speculative feature proposal. Do not infer independent confirmation merely
because two downstream objects have different names. Trace common inputs,
shared transformations, and inherited constraints.

## 1. System Objective And Non-Objectives

Trade Bot is a local, long-only, human-reviewed research and decision-support
system for swing/momentum allocation. It is intended to help a user with a
roughly 15-year accumulation horizon decide whether to maintain, reduce, or
increase exposure; compare strategy candidates; monitor frozen paper cohorts;
and preserve an auditable decision/execution record.

The system is not:

- an autonomous broker or order router;
- an intraday or high-frequency strategy;
- a claim that exact index levels or crashes can be forecast;
- a single end-to-end machine-learning model;
- a guarantee that backtested CAGR or drawdown will repeat;
- a license to treat news entered by the user as independent market evidence.

The default execution cadence is weekly Wednesday rebalancing (`W-WED`) with a
one-trading-day signal lag and configured transaction costs. Human review is
required before real action. The dashboard is primarily snapshot-backed so an
interactive page does not silently rerun expensive research with different
inputs.

## 2. High-Level Architecture

The two major paths are deliberately distinct:

```text
OPERATING PATH
market prices + macro + curated events + cached news
    -> configured strategy paths
    -> current-state diagnostics
    -> scenario lattice and narrative context
    -> allocation-authority gates
    -> absolute portfolio-risk constraints
    -> causal attribution + counterfactuals
    -> saved snapshot
    -> dashboard / human review / journal

RESEARCH PATH
fixed strategy or candidate definition
    -> backtest with lag, cadence, and costs
    -> rolling/regime/start-date/execution stress
    -> PBO, bootstrap, synthetic crash, ablation
    -> retrospective research status
    -> frozen prospective monitoring cohort
    -> possible later promotion decision
```

Primary implementation areas:

| Concern | Canonical implementation |
| --- | --- |
| Configuration and authority | `src/trade_bot/config.py`, `configs/baseline.yaml` |
| Strategy target weights | `src/trade_bot/strategies/` |
| Backtest execution | `src/trade_bot/backtest/engine.py` |
| Current market state | `src/trade_bot/research/current_state.py` |
| Scenario lattice | `src/trade_bot/research/future_scenarios.py` |
| Event and news context | `event_risk.py`, `news_monitor.py` |
| Final operating decision | `src/trade_bot/research/trade_decision.py` |
| Portfolio constraints | `src/trade_bot/portfolio/risk.py` |
| Snapshots and retention | `src/trade_bot/storage/run_store.py` |
| Operating history | `src/trade_bot/research/operating_history.py`, `storage/warehouse.py` |
| Defensive calibration | `defensive_judgement.py`, `defensive_layer_calibration.py` |
| Pre-break hindsight | `prebreak_hindsight.py`, `risk_policy_backtest.py` |
| V2.2 adversarial research | `i111_adversarial_validation.py`, `i111_execution_hardening.py` |
| Dashboard | `src/trade_bot/dashboard_v2/` |

DuckDB stores manifests and normalized operating tables. Large `BaselineRun`
objects are local pickle artifacts. Research reports are CSV/JSON/Markdown
artifacts with manifests where supported. This is a private/local design, not a
multi-user service boundary.

## 3. Exact Causal Authority Chain

The causal chain must not be summarized as “all inputs vote on risk.” The active
chain is sequential and gated:

```text
base strategy weights
    -> quantitative risk-status multiplier
    -> scenario probability multiplier * configured authority
    -> event/news multiplier * configured authority
    -> accepted macro multiplier * configured authority
    -> hard portfolio constraints
    -> decision-sanity governance cap
    -> final weights
```

Each stage stores its resulting defensive weight and marginal defensive
percentage-point addition. Later stages see the output of earlier stages, so
marginal attribution is path-dependent. It is a causal decomposition of the
implemented sequence, not a Shapley decomposition over every possible ordering.

### 3.1 Active authority configuration

The July 21 `balanced_asymmetric` policy is:

| Layer | Authority | Operating interpretation |
| --- | ---: | --- |
| Native/base market strategy | 1.00 | Sets the initial asset weights. |
| Quantitative risk status | 1.00 | Can reduce native risk exposure. |
| Scenario sizing | 0.00 | Probabilities are visible but cannot size. |
| Scenario portfolio budget | 0.00 | Scenarios cannot tighten hard limits. |
| Scenario-weighted stress | 0.00 | Advisory watch only. |
| Event/news sizing | 0.00 | Narrative layer is informational only. |
| Macro quantitative | 1.00 in config | Only an empirically accepted macro category can act; current categories add zero. |
| Absolute portfolio risk | 1.00 | Hard non-scenario limits can constrain. |
| Decision sanity | 1.00 | Governance guardrail, not a forecast. |

Scenario authority is fail-closed. If calibration status is `not_evaluated` or
`insufficient`, nonzero scenario authority is invalid configuration. Changing a
report file does not itself grant authority; a reviewed configuration change is
required.

### 3.2 Current-state risk score

Let `s_i` be the confirmation score for signal `i`, normally in {-1, 0, 1}.
The initial score is:

```text
r0 = 0.5 - mean(s_i) / 2
```

Add 0.10 if SPY drawdown is below -8%, add 0.10 if QQQ drawdown is below -10%,
add 0.10 when HYG momentum is bearish, and add 0.15 when VIXY momentum is
bullish. Clip the result to [0, 1]. Despite older prose occasionally using
“higher is constructive,” the code semantics are higher equals more risk.

Status thresholds and sizing multipliers are:

| Risk score | Status | Multiplier |
| ---: | --- | ---: |
| `< 0.25` | green | 1.00 |
| `[0.25, 0.45)` | yellow | 0.90 |
| `[0.45, 0.65)` | orange | 0.65 |
| `>= 0.65` | red | 0.40 |

This is a discontinuous mapping. A reviewer should explicitly test boundary
sensitivity near 0.25, 0.45, and 0.65 and compare the step function with a
monotone continuous alternative. Any alternative must be evaluated as a fixed
rule rather than chosen after seeing the best result.

### 3.3 Scenario probability transform

For the one-month scenario lattice, define probabilities `p_off`, `p_transition`,
and `p_fragile`. The raw scenario multiplier is:

```text
m_raw = clip(
    1 - 0.55*p_off - 0.20*p_transition - 0.15*p_fragile,
    0.40,
    1.00,
)
```

Configured sizing authority `a` produces:

```text
m_effective = 1 - a*(1 - m_raw)
```

Under the active policy `a = 0`, therefore `m_effective = 1` regardless of the
probability map. The raw number remains visible for diagnosis. This distinction
is essential: showing a 0.76 raw multiplier does not mean the portfolio used it.

### 3.4 Sequential risk budget

Before portfolio constraints, the engine uses the minimum active multiplier
across risk status, scenario, event, and accepted macro layers. Risk assets are
scaled by that multiplier; freed weight goes to the configured defensive asset,
currently BIL. The displayed final risk-budget multiplier is not merely the
minimum input multiplier. It is recomputed from the actual weights:

```text
final_budget = final_risk_asset_weight / base_risk_asset_weight
```

and clipped to [0, 1]. This avoids claiming a budget that does not match the
final target after portfolio constraints or governance.

### 3.5 Portfolio constraint split

The active utility profile has absolute limits including:

- maximum non-defensive single-asset weight 55%;
- maximum equity beta 1.05;
- maximum AI beta 0.85;
- expected shortfall 95% limit 3.5%;
- maximum named stress loss 18%;
- concentration HHI 0.42 as a watch rather than a hard block;
- scenario-weighted stress limit 8% as a watch because its authority is zero;
- maximum-turnover and correlation-shift diagnostics as soft/watch fields.

BIL is exempt from the risk-asset single-position cap. Constraint comparisons
use numeric tolerances so floating-point dust does not create a breach. Scenario
probabilities may still appear in diagnostic stress tables but cannot tighten
limits or force sizing while their authorities are zero.

## 4. July 21, 2026 Operating State

Snapshot identity:

```text
run_id: 20260721T223437.000001Z-31e37291-fda65c5b
market_date: 2026-07-21
config fingerprint prefix: 31e37291
prices: 5,420 rows x 169 columns
macro: 102 columns
configured strategies: 22
```

The base strategy is
`i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145`.

### 4.1 Current decision

| Quantity | Value |
| --- | ---: |
| Risk score/status | 0.433333 / yellow |
| Recommended action | HOLD |
| Base defensive weight | 59.6967% |
| Base risk-asset weight | 40.3033% |
| Quantitative status addition | 4.0303 percentage points defense |
| Scenario addition | 0.0000 pp |
| News/event addition | 0.0000 pp |
| Macro addition | 0.0000 pp |
| Portfolio hard-risk addition | 0.0000 pp |
| Decision-sanity addition | 0.0000 pp |
| Final defensive weight | 63.7270% |
| Final risk-asset weight | 36.2730% |
| Final risk-budget multiplier | 0.90 |

Rounded base weights are BIL 60%, SOXX 12%, SMH 11%, QQQ 11%, and AMZN 6%.
Rounded final weights are BIL 64%, SOXX 11%, SMH 10%, QQQ 10%, and AMZN 5%.

The portfolio risk engine reports `within_limits`, no applied hard constraints,
ES95 1.52%, maximum stress loss 12.70%, equity beta 0.755, and AI beta 0.453.
Scenario-weighted stress is 11.47% against an 8% advisory level, but it is a
watch rather than a hard constraint under zero authority. Concentration HHI is
also a watch. These watches must not be described as causes of the final target.

### 4.2 Current scenario map

The one-month probabilities in the decision record include:

- risk-off 24.91%;
- transition 38.56%;
- fragile upside 16.93%;
- broad risk-on 19.60%;
- constructive composite 28.06%.

The raw scenario formula implies a 0.7605 multiplier, but its effective
multiplier is 1.0 because sizing authority is zero. The system is permitted to
say the scenario map is cautious. It is not permitted to say that map reduced
today's weights.

### 4.3 News circularity counterfactual

The permanent counterfactual table contains active policy, news disabled, news
visible/informational-only, and a research-only news-sizing-enabled run. For
the first three operationally relevant cases, all reported quantities are
identical: risk score 0.4333, budget 0.90, scenario probabilities unchanged,
target BIL 64% / SOXX 11% / SMH 10% / QQQ 10% / AMZN 5%, equity beta 0.755,
and beta-adjusted SPY delta -0.0841.

Even the research-only news-authority counterfactual is identical on this date
because current event pressure is zero. This answers today's circularity test:
the final recommendation does not reflect the supplied AI, private-credit,
Iran, OpenAI, or Anthropic narratives through sizing. Those stories can still
shape a human reader, so UI language must keep “context” separate from “cause.”

## 5. Historical Snapshot Reconstruction And Provenance

After the authority repair, every retained historical daily-readout snapshot was
replaced one-for-one:

| Store | Before | Canonical dates | After | Range |
| --- | ---: | ---: | ---: | --- |
| Main operating snapshots | 222 | 222 | 222 | 2007-07-11 to 2026-07-21 |
| Pre-break snapshots | 440 generations | 397 | 397 | 2006-10-11 to 2025-03-21 |

The 43 extra pre-break rows were duplicate generations, not distinct market
dates. Keeping them would have created an accidental weighting hazard in any
analysis that failed to deduplicate.

The replacement did not reconstruct historical news using today's cache. It
loaded each recorded point-in-time snapshot, preserved its prices, macro,
events, news, strategy outputs, and diagnostics, and reapplied the new
allocation and portfolio policy. This is both faster and causally safer than
injecting today's narrative set backward. Old generations were retained until
all replacement dates were covered, then pruned to one per date. The command is
resumable by current config fingerprint.

The normalized operating history was separately rebuilt: 290 metric rows,
2,610 component rows, 3,190 scenario-driver rows, and 4,640 driver-rotation rows
from 2021-06-23 through 2026-07-21. It is explicitly labeled reconstructed
price-fast point-in-time history and is not misrepresented as prospective
monitoring.

Important limitation: pickle artifacts use the current Python class definitions
when loaded. Manifest hashes identify configs, not a cryptographic hash of every
source-code file. A stronger future design would record code commit/tree hash,
schema migration version, dependency lock hash, and input-data hashes for every
snapshot.

## 6. Scenario Probability Calibration

The probability audit uses 1,020 weekly point-in-time origins. Risk-off
probability is scored against matured SPY-versus-BIL and drawdown outcomes. It
reports Brier score/skill, log loss, AUC, expected calibration error, reliability
bins, block-bootstrap intervals, and expanding-history authority.

| Horizon | N | Positive rate | Mean predicted | Brier skill | AUC | ECE | Earned authority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 week | 998 | 41.88% | 27.68% | -0.0956 | 0.538 | 14.42% | 0.00% |
| 1 month | 995 | 41.51% | 27.69% | -0.0431 | 0.604 | 13.82% | 0.00% |
| 3 months | 986 | 38.13% | 27.72% | 0.0175 | 0.651 | 10.66% | 9.60% diagnostic result |

One- and three-month confidence intervals for Brier skill cross zero; one-week
skill is clearly negative. The configured operating horizon is one month, so
authority remains zero and status is `insufficient`. The 3-month result does
not automatically transfer to a 1-month decision.

The model tends to underpredict realized risk-off frequency in several bins.
For example, the 1-month 0.2-0.3 prediction bin has mean prediction 24.74% but
realized frequency 40.90%. That is meaningful calibration evidence, yet the
overall Brier skill remains negative because discrimination and calibration
must beat climatology jointly.

Review questions:

- Is the target definition economically aligned with the use of the probability?
- Does calibration improve with isotonic/logistic recalibration fit only on
  prior origins?
- Are probability bins stable across eras and volatility regimes?
- Does the current-universe feature set create survivorship leakage?
- Should the 3-month model be a separate research object rather than an
  argument for reviving one-month sizing?

## 7. “Correctly Defensive?” Results

Two distinct audits answer different questions.

### 7.1 Strategy-native defense

This audit measures the base strategy's effective BIL plus residual cash. At a
65% defensive threshold for the focus strategy:

| Horizon | Episode starts | Correct | False alarm | Mixed | Median forward SPY drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 week | 46 | 47.8% | 39.1% | 13.0% | -1.6% |
| 1 month | 44 | 47.7% | 29.5% | 22.7% | -4.2% |
| 3 months | 44 | 43.2% | 29.5% | 27.3% | -7.0% |

“Correct” means SPY lagged BIL or suffered the horizon-specific drawdown;
“false alarm” means SPY materially beat BIL without the drawdown; the remainder
is mixed/early. These labels encode a utility choice and should be sensitivity
tested. They do not mean a 47.7% probability of a crash.

### 7.2 Active layered policy

The active study uses 1,020 weekly origins and materiality thresholds of base
defense at least 55%, quantitative addition at least 5 percentage points, and
portfolio addition at least 1 point. Scenario additions are separately recorded
and equal zero at every origin.

No origin had all three active material layers simultaneously. This is not
missing data. When base plus quantitative defense was already high, hard
portfolio constraints did not require another material clamp. Thus an
“all-three agreement predicts crashes” claim is not estimable for the revised
policy.

Incremental episode comparisons:

| Comparison | Horizon | Left/right starts | Delta correct | Delta false alarm | Median return cost | Median DD improvement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base + quantitative vs base only | 1w | 50 / 40 | +6.0 pp | -30.5 pp | -0.18% | +0.54% |
| Base + quantitative vs base only | 1m | 50 / 40 | +2.5 pp | -7.0 pp | -0.60% | +1.20% |
| Base + quantitative vs base only | 3m | 50 / 39 | +11.5 pp | -1.9 pp | -1.96% | +2.63% |
| Quantitative + portfolio vs quantitative only | 1w | 69 / 87 | +6.4 pp | -16.4 pp | +0.02% | +0.03% |
| Quantitative + portfolio vs quantitative only | 1m | 69 / 87 | -1.4 pp | +5.2 pp | -0.48% | +0.21% |
| Quantitative + portfolio vs quantitative only | 3m | 67 / 85 | -3.6 pp | +1.3 pp | -0.43% | +0.05% |

Interpretation: price-derived risk status adds some historical downside
discrimination to native defense. Portfolio clamps add short-horizon safety but
do not improve one-to-three-month forecasting discrimination. The layers share
price inputs; they are distinct causal pathways, not statistically independent
votes.

The predeclared 60% base-defense sensitivity strengthens rather than reverses
the base-plus-quantitative result. Relative to base-only episodes, its 44 versus
33 one-month starts improve correct defense by 6.8 points and reduce false
alarms by 17.4 points; its 44 versus 32 three-month starts improve correct
defense by 12.5 points and reduce false alarms by 11.9 points. Median return
costs are 0.53% and 1.79%, with drawdown improvements of 1.24% and 2.25%.
Portfolio additions remain weak at one and three months. This threshold
sensitivity is encouraging for the price-status interaction, but the cohorts
are not randomized and the threshold family is still researcher-chosen.

### 7.3 Opportunity-cost replay

Non-overlapping weekly policies from 2007-05-30 through 2026-07-21:

| Policy | CAGR | Max DD | Terminal wealth | Delta CAGR | DD improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base weekly | 20.64% | -24.82% | 36.29 | — | — |
| Quantitative sizing every week | 15.16% | -19.74% | 14.90 | -5.48 pp | +5.09 pp |
| Full hard-risk path every week | 12.56% | -16.59% | 9.62 | -8.08 pp | +8.23 pp |
| Current-configuration-only overlay | 20.49% | -24.82% | 35.43 | -0.15 pp | 0.00 pp |

The last row is the closest answer to “is the small price for patience actually
small?” Historically its cost was small, but it did not improve the full-period
maximum drawdown. This supports modest patience, not a claim that 63.73% BIL is
the uniquely optimal weight.

## 8. Pre-Break Hindsight And Sparse Policy Replay

The pre-break panel combines canonical event-window snapshots with ordinary
reference controls: 485 deduplicated analyzed observations from 2006-10-11 to
2026-07-21, including 42 post-break event-window snapshots. The three-month
severe-label share is 35.0% and major-label share 14.5%.

Top hindsight associations include energy/inflation relief, cycle acceleration,
credit pressure, leadership acceleration, cross-sectional dispersion,
pre-break probability, dollar pressure, and QQQ three-month return. These are
ranked retrospective associations and should generate purged/fixed tests; they
are not deployable rules.

After causal attribution repair, early hard-defense sources are:

- early-watch: quantitative risk status 95.5%, base already defensive 4.5%;
- long-lead context: quantitative risk status 82.9%, base already defensive
  16.2%, portfolio absolute risk 1.0%;
- scenario probabilities and news/events: zero causal additions under the
  replacement policy.

The snapshot-budget policy replay overlays sparse historical readouts on eight
selected experiment strategies. Base median CAGR is 14.99% with median max DD
-24.58%. The actual snapshot budget lowers median CAGR to 12.24% and does not
improve median max DD. Hindsight stage floors lose less CAGR but still do not
improve the median max drawdown. Because event windows are sparse and some
variants use hindsight stage knowledge, this report is best used to reject
aggressive early defense, not to select a live floor.

The lead-time tradeoff is severe. Delaying hard defense until 15, 21, 30, or 45
days before the break raises candidate budgets but misses approximately 73.6%,
66.2%, 53.2%, or 40.5% of severe labels, respectively, while pre-trigger false
alarm shares remain about 31-35%. There is no clean timing threshold in this
sample.

## 9. Primary Strategy Evidence And Fragility

The focus strategy is historically attractive but not promotion-ready. The
important evidence is deliberately contradictory.

### 9.1 Favorable retrospective evidence

- Primary path roughly 22% CAGR and -20% max drawdown under configured Wednesday
  execution.
- Candidate-family PBO estimate 1.43% with 0% OOS-loss probability in the fixed
  candidate study; this addresses within-family selection risk, not the full
  historical research process.
- Five-year block-bootstrap median annualized return about 22.0% for the focus
  strategy, p05 about 9.1%.
- Native defense shows nontrivial but imperfect downside discrimination.
- Carried-state start-date variants did not produce negative minimum CAGR in the
  evaluated set.

### 9.2 Adverse evidence

- Daily rebalance: roughly 19.6% CAGR and -28.9% drawdown.
- Monday rebalance: roughly 19.1% and -29.9%.
- Execution stress generated 74 failure rows in the adversarial suite.
- Five-year bootstrap probability of breaching -25% drawdown is about 15% for
  the focus strategy.
- Synthetic AI-crash p05 historical-weight stress is about -31%; current stress
  about -11.5%.
- Average AI/growth exposure is about 65.5%, so the strategy is not independent
  of the leadership thesis it is meant to manage.
- AI warning monitors have substantial false-positive rates near 43-45%.
- Clean AI-led historical break events are scarce.

### 9.3 Failed repair hypotheses

No V2.2 hardening mechanism cleared the retrospective promotion screen.
Risk-sleeve AI caps either failed to repair the worst path or sacrificed too
much return. Hold/step rules improved drawdown but became tradeoff-only.

A fixed, non-grid cross-sectional replacement test deferred new/increased AI
targets when six of eight stress components agreed. Sending blocked weight to
BIL produced 18.85% Wednesday CAGR, -26.69% Wednesday drawdown, -30.32% worst
execution drawdown, and eight failures. Sending it to RSP produced 19.29%,
-27.60%, -34.29%, and eight failures. The reference had 21.01%, -20.34%,
-30.60%, and seven failures. Both hypotheses were rejected.

Fixed execution smoothing reduced weekday dispersion but did not clear all
gates. EWM5 produced 20.62% Wednesday CAGR and -23.73% drawdown; mean10 produced
20.19% and -24.57%. Both remain research-only.

The correct operating label is `promising_but_fragile`. Approximately 22% CAGR
/-20% drawdown is a configured-path observation, not an expected live result.

## 10. Data Integrity, Leakage, And Independence Audit

### 10.1 Controls that exist

- Strategy execution uses a one-day signal lag by default.
- Historical snapshot replacement preserves recorded point-in-time narrative
  inputs rather than replaying today's news backward.
- Scenario authority is earned through expanding-history calibration and gated
  explicitly.
- News cache fallback is triage-only and carries source/as-of health semantics.
- Curated/news events can decay and expire.
- Pre-break panels deduplicate market dates and separate event experiments from
  ordinary controls.
- Historical origin features are truncated or causally sliced; an equivalence
  test compares sliced full paths with true truncated recomputation.
- Research reports expose survivorship, pre-inception, and sample-size caveats.
- Prospective monitoring cohorts freeze strategy and execution definitions.

### 10.2 Remaining risks

- Current-universe survivorship: many studies replay today's tickers backward.
- Pre-inception proxies: newer ETFs and AI assets have short histories.
- Researcher degrees of freedom: PBO for one candidate shelf does not count all
  prior abandoned ideas.
- Event selection hindsight: named crises and break dates are known today.
- Overlapping horizons: episode outcomes can be serially dependent.
- Shared market inputs: base strategy, risk status, scenarios, and portfolio risk
  are not independent even when their transformations differ.
- Threshold discontinuity: small score changes can cause large multiplier jumps.
- Snapshot code provenance: config hashes are stronger than no provenance but do
  not fully identify code and data state.
- Pickle compatibility/security: artifacts are local/trusted-only and coupled to
  Python class evolution.
- External macro revision risk: FRED histories may contain revised values unless
  vintages are explicitly frozen.
- Yahoo-style adjusted-price behavior and delisting coverage can bias results.
- Multiple reports may use different units of analysis: daily observations,
  weekly origins, episode starts, fixed forward windows, or policy paths.
- Human circularity remains possible even if code authority is zero: a narrative
  can influence the operator who then interprets quantitative caution more
  strongly.

### 10.3 Independence taxonomy

Use these labels instead of a binary “independent/not independent” statement:

| Pair | Relationship |
| --- | --- |
| User-supplied news vs final weights | Causally disconnected under active zero authority. |
| Scenario probabilities vs final weights | Causally disconnected under active zero authority. |
| Native strategy vs risk status | Distinct algorithms, shared market-price history. |
| Risk status vs portfolio hard risk | Distinct transformation/constraint, shared prices and candidate weights. |
| Scenario vs portfolio watch metrics | Shared scenario lattice diagnostically; zero sizing authority. |
| Pre-break labels vs event selection | Hindsight-dependent by construction. |

## 11. Dashboard And Human-Factors Semantics

The V2.2 dashboard is summary-first and snapshot-backed. Operating, Research,
Simulation, Monitoring, Risk, and Macro workbenches load heavy diagnostics only
when requested. The daily surface should maintain these language rules:

- “caused,” “reduced,” or “added defense” only for nonzero persisted marginal
  attribution;
- “watch,” “context,” or “diagnostic” for zero-authority scenario/news fields;
- “within limits” must not be conflated with “low risk”;
- a raw scenario multiplier must be shown beside effective authority if shown at
  all;
- `HOLD` means no material change from the systematic target, not “safe” or
  “bullish”;
- historical correctness rates must include episode count and false-alarm rate;
- retrospective, reconstructed, and prospective evidence must be visually
  distinct.

The event selector in the pre-break research page belongs inside the selected
event analysis section, not visually before its parent heading. UI layout is not
merely cosmetic here: misplaced controls can imply a global operating effect
when they actually filter a hindsight research view.

## 12. Validation And Reproducibility Map

Key commands:

```bash
poetry run trade-bot build-snapshot --config configs/baseline.yaml
poetry run trade-bot seed-operating-history --config configs/baseline.yaml
poetry run trade-bot audit-defensive-judgement
poetry run trade-bot calibrate-defensive-layers --config configs/baseline.yaml
poetry run trade-bot calibrate-scenario-probabilities --config configs/baseline.yaml
poetry run trade-bot analyze-prebreak-hindsight
poetry run trade-bot build-research-governance-ledger
poetry run pytest -q
```

Canonical evidence artifacts:

| Question | Artifact |
| --- | --- |
| Current decision and attribution | latest snapshot in `data/run_store/snapshots/` |
| Strategy-native correctness | `reports/defensive_signal_audit/` |
| Active layer correctness and regret | `reports/defensive_layer_calibration/` |
| 60% base-threshold sensitivity | `reports/defensive_layer_calibration_base60/` |
| Scenario calibration | `reports/scenario_probability_calibration/` |
| Historical pre-break behavior | `reports/prebreak_hindsight/` |
| Sparse policy utility | `reports/prebreak_risk_policy_backtest/` |
| i111 adversarial evidence | `reports/i111_adversarial_validation/` |
| Execution fragility | `reports/i111_execution_hardening/` |
| Fixed rejected repairs | `reports/i111_cross_sectional_replacement/`, `reports/i111_execution_smoothing/` |
| Prospective evidence | DuckDB monitoring windows with frozen start dates |

Do not claim “full simulation validation” unless the long rolling-origin job has
completed and written its expected artifacts. A process that ran for hours but
produced no final artifact is incomplete evidence.

## 13. Current System Read, With Confidence Levels

High confidence:

- News and user-supplied narratives did not cause the July 21 target.
- Scenario probabilities did not cause the July 21 target.
- The 77% defensive result was partly a structural consequence of retired
  scenario/portfolio authority and should not be used.
- The base strategy itself is meaningfully defensive today.
- The leading strategy is execution-fragile and AI/growth-concentrated.
- Aggressive continuous defensive overlays have historically sacrificed large
  amounts of terminal wealth.

Moderate confidence:

- Price-derived risk status adds some incremental downside discrimination to
  native strategy defense.
- A small 10% reduction of the remaining risk sleeve is a defensible patience
  adjustment, especially when information value is high.
- Hard portfolio constraints are useful safety controls but not independent
  medium-horizon forecasts.
- Current concentration/dispersion conditions justify avoiding emotional rally
  chasing even though credit and volatility do not confirm a break.

Low confidence:

- 63.73% is the uniquely correct defensive allocation.
- Current caution predicts an imminent crash.
- The historical Wednesday i111 path will survive live execution.
- Any existing AI warning threshold is ready for allocation authority.
- Scenario probabilities deserve nonzero one-month sizing authority.

## 14. Highest-Value Independent Review Questions

Please answer these in priority order and distinguish bugs from research ideas.

1. Does the implemented causal attribution omit any hidden path by which news,
   events, or scenarios can alter base strategy weights, current-state scores,
   macro acceptance, or portfolio inputs before the explicit authority gate?
2. Is the risk-score orientation and documentation consistent everywhere? Find
   any location that still says higher score is more constructive.
3. Are the step multipliers 1.00/0.90/0.65/0.40 calibrated, or merely plausible?
   Propose one fixed, low-degree-of-freedom test of a smoother mapping.
4. Does the layered episode comparison condition on post-treatment variables or
   create collider bias by grouping on multiple downstream clamps?
5. Is “correct defense” economically well specified for a 15-year contributor,
   or should regret be contribution-weighted, tax-aware, and recovery-aware?
6. Does the frozen-weight episode method overstate or understate realistic
   regret relative to the strategy's actual rebalancing/re-entry behavior?
7. Why does the full weekly policy improve drawdown but destroy so much CAGR?
   Decompose signal timing, time out of market, turnover, and concentration.
8. Can the small current-configuration result be replicated with purged eras,
   alternate thresholds, and block-bootstrap uncertainty without selecting the
   configuration after seeing today?
9. Are portfolio ES and stress estimates stable to lookback, covariance
   shrinkage, fat tails, and correlated AI shocks?
10. Is BIL an adequate defensive proxy across inflation/rates regimes? Test
    cash, intermediate Treasuries, gold, and mixtures without turning the test
    into a large optimization grid.
11. How much of i111's edge is Wednesday close-to-close timing, data timestamp
    convention, or rebalance-calendar luck?
12. Can execution fragility be reduced through a causal mechanism that preserves
    the return engine, rather than smoothing or capping it bluntly?
13. Are candidate PBO estimates understated because the candidate shelf omits
    prior research trials? Suggest a research-wide trial accounting method.
14. Which historical inputs are revised macro series rather than true vintages,
    and how large could vintage leakage be?
15. Are delisted assets and point-in-time ETF availability handled honestly in
    every report? Identify claims that should be downgraded if not.
16. Does prospective monitoring currently freeze code/config/data sufficiently
    to constitute a real forward test?
17. Are UI phrases capable of reintroducing narrative circularity even when the
    sizing code is clean?
18. What one or two next experiments have the highest expected information gain
    and the lowest researcher degrees of freedom?

## 15. Preferred Form Of External Feedback

Return feedback in this structure:

1. `Critical implementation defects`: exact file/function, causal consequence,
   minimal reproduction, and proposed test.
2. `Methodological invalidators`: leakage, dependence, sample construction, or
   target-definition issues that would overturn a result.
3. `Claims that are too strong`: quote/paraphrase, evidence actually available,
   and corrected wording.
4. `Robust findings`: conclusions that survive the audit and why.
5. `Fixed next experiments`: at most five, with predeclared hypothesis,
   treatment, comparator, metric, gate, and stopping rule.
6. `Do not pursue`: attractive ideas likely to add overfit, complexity, or
   narrative feedback without enough information gain.

Avoid generic suggestions such as “use more machine learning,” “add sentiment,”
or “optimize the weights.” Any proposed model must identify its target, causal
availability time, training protocol, comparator, costs, failure condition, and
authority gate.

## 16. Bottom Line

Trade Bot is strongest as a transparent research and decision-support system,
not as an oracle. The repaired July 21 result is mostly the native strategy's
own 59.70% defensive posture plus a four-percentage-point price-state adjustment.
News and scenarios add zero. Hard portfolio constraints are satisfied without a
clamp. Historical evidence says native and quantitative agreement contains some
downside information, but continuous layered defense is expensive and the
precise 63.73% target is not proven optimal.

The most important unresolved issue is no longer narrative circularity in the
implemented weight calculation. It is whether a high-growth, AI-concentrated,
execution-fragile strategy can retain enough of its historical return engine
under real timing, costs, and future leadership regimes. The proper next step is
frozen prospective evidence and a small number of adversarial, predeclared
tests—not another broad parameter search.
