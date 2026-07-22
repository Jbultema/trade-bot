# Trade Bot V2.2 AI Review Whitepaper

Status: independent-review packet. Evidence cut: 2026-07-22. This document is
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
close-safe two-row signal shift and configured transaction costs. A target using
close `t` is modeled as filled at close `t+1` and first earns the next
close-to-close interval. Human review is
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

The Research path now has an explicit comparison contract. A scorecard is
eligible for the shared frontier only when it belongs to the complete canonical
library and shares one hash over:

- exact price-frame values and ordered columns;
- start and market dates;
- close-to-close return convention and 252-session annualization;
- close observation, first-eligible fill, signal lag, rebalance cadence, and
  transaction costs;
- outcome-planning assumptions: $220,000 start, $4,000 annual contributions
  deposited monthly, 15 years, and the soft/hard drawdown bands;
- full configuration, dependency lock, and research source tree.

The source archive preserves the exact serialized candidate definitions. Replay
loads those definitions directly; it does not regenerate or reselect them. A
root manifest enumerates every expected iteration, candidate count, candidate
roster hash, iteration-manifest hash, and artifact-integrity result. It is
written last. Dashboard loaders reject a missing, partial, stale, config-mismatched,
or byte-mismatched library. Configured strategies are included in the same replay
and live-snapshot scorecards are only a fallback when no canonical library is
available, so they cannot silently supersede comparable rows.

The completed contract contains 427 scorecard rows across 73 groups: 406 exact
replays of the archived iteration 77-164 definitions and 21 configured runtime
strategies. The 406-row one-to-one migration changed median CAGR by +0.003
percentage points and median maximum drawdown by -0.237 points. The full ranges
were -2.160 to +1.967 CAGR points and -7.001 to +10.053 drawdown points. Despite
the near-zero aggregate center, 114 promotion labels changed and the top-20
membership turned over by nine names. A contribution-only counterfactual on the
new return paths changed 223 raw rank positions but zero top-20 members; many of
those lower-order moves are tie ordering. Thus the $70,000-to-$4,000 correction
is highly material to terminal-wealth dollars, modest to normalized utility, and
not evidence for a different champion.

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
    -> calibration-gated quantitative risk-timing multiplier
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
| Quantitative risk timing | 0.00 | Visible research state; failed the promotion gate and cannot size. |
| Scenario sizing | 0.00 | Probabilities are visible but cannot size. |
| Scenario portfolio budget | 0.00 | Scenarios cannot tighten hard limits. |
| Scenario-weighted stress | 0.00 | Advisory watch only. |
| Event/news sizing | 0.00 | Narrative layer is informational only. |
| Macro quantitative | 0.00 | Revised-history FRED inputs are descriptive only; nonzero authority requires calibrated point-in-time or first-release vintages. |
| Absolute portfolio risk | 1.00 | Hard non-scenario limits can constrain. |
| Decision sanity | 1.00 | Governance guardrail, not a forecast. |

Scenario authority is fail-closed. If calibration status is `not_evaluated` or
`insufficient`, nonzero scenario authority is invalid configuration. Changing a
report file does not itself grant authority; a reviewed configuration change is
required.

Risk-timing authority is independently fail-closed. The replacement distinguishes
fragility from confirmed credit, volatility, breadth, and trend deterioration,
but its 1,020-origin replay did not improve risk-adjusted return decisively. It
therefore remains at zero authority with `insufficient` calibration status. Full rule and replay
details are in `docs/risk_timing_research.md`.

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

## 4. July 22, 2026 Operating State

Snapshot identity:

```text
run_id: 20260722T154729.000003Z-b1df1c1e-5d839408
market_date: 2026-07-22
config fingerprint prefix: b1df1c1e
prices: 5,421 rows x 169 columns
macro: 102 columns
configured strategies: 22
```

The base strategy is
`i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145`.

### 4.1 Current decision

| Quantity | Value |
| --- | ---: |
| Risk score/status | 0.400000 / yellow |
| Recommended action | HOLD |
| Base defensive weight | 57.8017% |
| Base risk-asset weight | 42.1983% |
| Quantitative timing addition | 0.0000 percentage points defense |
| Risk-timing raw/effective multiplier | 1.00 / 1.00 at 0% authority |
| Scenario addition | 0.0000 pp |
| News/event addition | 0.0000 pp |
| Macro addition | 0.0000 pp |
| Portfolio hard-risk addition | 0.0000 pp |
| Decision-sanity addition | 0.0000 pp |
| Final defensive weight | 57.8017% |
| Final risk-asset weight | 42.1983% |
| Final risk-budget multiplier | 1.00 |

Rounded base and final weights are BIL 58%, SOXX 13%, SMH 12%, QQQ 11%,
and AMZN 6%.

The portfolio risk engine reports `within_limits`, no applied hard constraints,
ES95 1.77%, maximum stress loss 14.77%, equity beta 0.896, and AI beta 0.527.
Scenario-weighted stress is 13.16% against an 8% advisory level, but it is a
watch rather than a hard constraint under zero authority. Concentration HHI is
also a watch. These watches must not be described as causes of the final target.

### 4.2 Current scenario map

The one-month probabilities in the decision record include:

- risk-off 22.83%;
- transition 37.87%;
- fragile upside 17.76%;
- broad risk-on 21.54%;
- constructive composite 30.42%.

The raw scenario formula implies a 0.7721 multiplier, but its effective
multiplier is 1.0 because sizing authority is zero. The system is permitted to
say the scenario map is cautious. It is not permitted to say that map reduced
today's weights.

### 4.3 News circularity counterfactual

The permanent counterfactual table contains active policy, news disabled, news
visible/informational-only, and a research-only news-sizing-enabled run. For
the first three operationally relevant cases, all reported quantities are
identical: risk score 0.4000, budget 1.00, scenario probabilities unchanged,
target BIL 58% / SOXX 13% / SMH 12% / QQQ 11% / AMZN 6%, equity beta 0.896,
and beta-adjusted SPY delta 0.0000.

The research-only news-authority counterfactual is different: its 8% raw event
pressure would reduce the budget to 0.92, raise BIL to 61%, and lower equity
beta to 0.824. That counterfactual is deliberately non-operational. The three
authorized cases answer today's circularity test: the final recommendation does
not reflect the supplied AI, private-credit, Iran, OpenAI, or Anthropic
narratives through sizing. Those stories can still shape a human reader, so UI
language must keep “context” separate from “cause.”

## 5. Historical Snapshot Reconstruction And Provenance

After the authority repair, every retained historical daily-readout snapshot was
replaced one-for-one:

| Store | Before | Canonical dates | After | Range |
| --- | ---: | ---: | ---: | --- |
| Main operating snapshots | 354 generations | 223 | 223 | 2007-07-11 to 2026-07-22 |
| Pre-break snapshots | 397 dates | 397 | 397 | 2006-10-11 to 2025-03-21 |

The current stores contain exactly one row per planned market date. An earlier
audit had found 43 extra pre-break generations; those duplicates were removed
before this complete replacement, eliminating the accidental weighting hazard.

The replacement did not reconstruct historical news using today's cache. It
loaded each recorded point-in-time snapshot, preserved its prices, macro,
events, news, strategy outputs, and diagnostics, and reapplied the new
allocation and portfolio policy. This is both faster and causally safer than
injecting today's narrative set backward. Old generations were retained until
all replacement dates were covered, then pruned to one per date. The command is
resumable by current config fingerprint.

The normalized operating history was separately rebuilt: 290 metric rows,
2,610 component rows, 3,190 scenario-driver rows, and 4,640 driver-rotation rows
from 2021-06-23 through 2026-07-22. It is explicitly labeled reconstructed
price-fast point-in-time history and is not misrepresented as prospective
monitoring.

Important limitation: snapshot payloads remain pickle files and therefore load
through the current Python class definitions. Their manifests now record git
and source-tree identity, dependency-lock and project hashes, config identity,
schema version, and the exact ordered price-frame hash. This detects stale or
mismatched evidence but does not provide language-independent archival storage;
schema-versioned Parquet/JSON remains the stronger long-term format.

## 6. Scenario Probability Calibration

The probability audit uses 1,020 weekly point-in-time origins. Risk-off
probability is scored against matured SPY-versus-BIL and drawdown outcomes. It
reports Brier score/skill, log loss, AUC, expected calibration error, reliability
bins, block-bootstrap intervals, and expanding-history authority.

| Horizon | N | Positive rate | Mean predicted | Brier skill | AUC | ECE | Earned authority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 week | 999 | 41.94% | 27.68% | -0.0963 | 0.538 | 14.48% | 0.00% |
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

| Horizon | Episode starts | Beneficial under rule | Costly false positive | Mixed | Median forward SPY drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 week | 43 | 53.5% | 37.2% | 9.3% | -1.5% |
| 1 month | 42 | 45.2% | 33.3% | 21.4% | -4.2% |
| 3 months | 42 | 42.9% | 31.0% | 26.2% | -6.9% |

“Correct” means SPY lagged BIL or suffered the horizon-specific drawdown;
“false alarm” means SPY materially beat BIL without the drawdown; the remainder
is mixed/early. These labels encode a utility choice and should be sensitivity
tested. They do not mean a 45.2% probability of a crash.

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
| Base + quantitative vs base only | 1w | 10 / 48 | +26.7 pp | -25.8 pp | -0.1% | +0.8% |
| Base + quantitative vs base only | 1m | 10 / 48 | +2.5 pp | -23.7 pp | -1.1% | +1.3% |
| Base + quantitative vs base only | 3m | 10 / 47 | +16.0 pp | -8.3 pp | -2.4% | +1.9% |
| Quantitative + portfolio vs quantitative only | 1w | 44 / 19 | +16.1 pp | -5.7 pp | -0.1% | +0.3% |
| Quantitative + portfolio vs quantitative only | 1m | 44 / 19 | -11.7 pp | +12.3 pp | -0.5% | +0.9% |
| Quantitative + portfolio vs quantitative only | 3m | 44 / 19 | -20.0 pp | +8.5 pp | -1.0% | +1.4% |

Interpretation: the legacy price-derived risk status showed some episode-level
downside discrimination, but its continuous sizing rule was too costly. The new
confirmation-timed candidate did not improve full-period maximum drawdown, so it
has zero allocation authority. Portfolio clamps add short-horizon safety but are
not one-to-three-month forecasts. The layers share price inputs; they are
distinct causal pathways, not statistically independent votes.

The confirmed-timing intersection has only 10 base-plus-quantitative episode
starts. Its lower false-alarm rate is interesting, but median return costs are
1.1% at one month and 2.4% at three months. Portfolio additions remain weak
at one and three months. These cohorts are not randomized, and the small
confirmed sample cannot justify authority.

### 7.3 Opportunity-cost replay

Non-overlapping weekly policies from 2007-05-30 through 2026-07-22:

| Policy | CAGR | Max DD | Terminal wealth | Delta CAGR | DD improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base weekly | 19.60% | -25.90% | 30.71 | — | — |
| Legacy risk-status sizing | 14.30% | -21.10% | 13.00 | -5.20 pp | +4.80 pp |
| Confirmation-timed candidate | 19.10% | -26.20% | 28.50 | -0.50 pp | -0.40 pp |
| Candidate plus hard-risk path | 14.50% | -20.80% | 13.33 | -5.10 pp | +5.10 pp |
| Current authorized overlay only | 19.50% | -26.20% | 30.19 | -0.10 pp | -0.30 pp |

The confirmation-timed row is the cleanest test of whether a smaller price for
patience buys protection. It cost 0.50 CAGR points while making maximum drawdown
0.40 points worse. The current authorized overlay is nearly neutral but also
does not improve drawdown. Neither result grants the timing layer sizing authority.

## 8. Pre-Break Hindsight And Sparse Policy Replay

The pre-break panel combines canonical event-window snapshots with ordinary
reference controls: 486 deduplicated analyzed observations from 2006-10-11 to
2026-07-22, including 42 post-break event-window snapshots. The three-month
severe-label share is 34.9% and major-label share 14.5%.

Top hindsight associations include energy/inflation relief, cycle acceleration,
credit pressure, leadership acceleration, cross-sectional dispersion,
pre-break probability, dollar pressure, and QQQ three-month return. These are
ranked retrospective associations and should generate purged/fixed tests; they
are not deployable rules.

After the timing-authority repair and 1:1 snapshot replacement, early hard-defense
sources are:

- early-watch: portfolio absolute risk 81.8%, base already defensive 18.2%;
- long-lead context: portfolio absolute risk 69.5%, base already defensive
  30.5%;
- quantitative timing, scenario probabilities, and news/events: zero causal
  additions under the replacement policy.

The refreshed snapshot-budget replay overlays sparse historical readouts on
eight selected experiment strategies. Base median CAGR is 13.09% with median
max DD -22.10%. The actual snapshot budget lowers median CAGR to 11.94% and does
not improve median max DD. Hindsight stage floors lose less CAGR but still do not
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

- The close-safe configured primary path produced 20.67% CAGR and -25.80% max
  drawdown; the native i111 challenger produced 20.84% and -24.75%.
- The active 12-candidate adversarial roster has a 21.43% family PBO estimate
  with 0% OOS-loss probability. This is roster-specific and does not override
  the broader canonical-shelf result below.
- Five-year block-bootstrap median annualized return is 16.89% for the focus
  strategy, with a 4.11% p05.
- Native defense shows nontrivial but imperfect downside discrimination.
- Carried-state start-date variants did not produce negative minimum CAGR in the
  evaluated set.

### 9.2 Adverse evidence

- Daily rebalance: 19.69% CAGR and -28.86% drawdown.
- Monday rebalance: 19.16% and -29.85%.
- Execution stress generated 84 failure rows in the active adversarial suite.
- The canonical 20-candidate shelf PBO is 64.29%, labeled high overfit risk,
  despite a 0% OOS-loss rate. PBO is materially roster-sensitive.
- Five-year bootstrap probability of breaching -25% drawdown is 24.36% for
  the focus strategy.
- Synthetic AI-crash p05 historical-weight stress is -30.88%; current stress
  is -11.94%.
- Average AI/growth exposure is 65.57%, so the strategy is not independent
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

The correct operating label is `promising_but_fragile`. The former 22.18% CAGR
/-19.68% drawdown result used a one-row close-boundary approximation and is not
an operating or expected-live result.

### 9.4 Contribution-aware catastrophic-tail utility

The fixed experiment uses 1,000 block-bootstrap paths, a 21-session block,
15-year horizon, $220,000 starting value, $4,000 annual contributions deposited monthly,
and seed `20260705`. Terminal wealth includes contributions. Drawdown and Ulcer
Index use a separate unitized return index, so cash flows cannot mechanically
hide a market loss.

| Path | P(DD > 20%) | P(DD > 25%) | P(DD > 30%) | Mean DD conditional on >20% | P(10%-path wealth target) | Terminal wealth p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Configured primary | 84.3% | 48.5% | 22.1% | -27.2% | 98.5% | $4.22M |
| Native i111 challenger | 83.7% | 47.8% | 21.7% | -27.1% | 98.6% | $4.29M |
| SPY hold | 99.1% | 92.3% | 73.8% | -36.9% | 60.9% | $1.25M |
| QQQ hold | 99.9% | 95.9% | 80.0% | -38.2% | 82.2% | $2.21M |

This supports the claim that the strategy historically transformed catastrophic
equity tails into smaller but still frequent drawdowns. It does not establish a
forward probability: the bootstrap resamples the same modern-universe replay
whose survivorship and selection validity remain blocked.

## 10. Data Integrity, Leakage, And Independence Audit

### 10.1 Controls that exist

- Strategy execution uses a close-safe two-row shift by default. The former
  one-row shift is explicitly labeled `close_boundary_approximation`.
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
- Snapshot manifests now record git commit, committed tree, dirty status and
  status hash, exact source-tree hash, dependency-lock and project hashes,
  schema version, and an exact price-frame identity. Replacement mode requires
  both matching config and matching source-tree identity before it can resume.
- Replacement mode recomputes price-derived strategy paths, performance/window
  metrics, strategy alerts, event outcome tables, and the final decision while
  preserving the recorded point-in-time narrative objects.

### 10.2 Remaining risks

- Current-universe survivorship: many studies replay today's tickers backward.
- Pre-inception proxies: newer ETFs and AI assets have short histories.
- Researcher degrees of freedom: the ledger currently indexes 569 manifested
  trial rows, but 113 artifact directories have no manifest and three of 19
  manifests lack explicit rosters. Candidate-shelf PBO does not count abandoned
  or unmanifested ideas, so retrospective promotion is disabled.
- Event selection hindsight: named crises and break dates are known today.
- Overlapping horizons: episode outcomes can be serially dependent.
- Shared market inputs: base strategy, risk status, scenarios, and portfolio risk
  are not independent even when their transformations differ.
- Threshold discontinuity: small score changes can cause large multiplier jumps.
- Legacy market caches may lack vendor fetch-time metadata even though the exact
  frame used by each new snapshot is hashed. Future refreshes write a Yahoo/
  yfinance sidecar with fetch time and known limitations.
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

- Explicit credit, volatility, breadth, and trend gates are more interpretable
  than the legacy aggregate status multiplier.
- Patient positioning can still come from the native strategy, but the separate
  timing candidate has not earned authority.
- Hard portfolio constraints are useful safety controls but not independent
  medium-horizon forecasts.
- Current concentration/dispersion conditions justify avoiding emotional rally
  chasing even though credit and volatility do not confirm a break.

Low confidence:

- 60.02% is the uniquely correct defensive allocation.
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
not as an oracle. The repaired July 21 cached-data result is the native
strategy's own 60.02% defensive posture. Revised-history macro, risk timing,
news, and scenarios add zero; hard portfolio constraints are satisfied without
a clamp. Historical evidence says continuous layered defense is expensive, and
the precise 60.02% target is not proven optimal.

The most important unresolved issue is no longer narrative circularity in the
implemented weight calculation. It is whether a high-growth, AI-concentrated,
execution-fragile strategy can retain enough of its historical return engine
under real timing, costs, and future leadership regimes. The proper next step is
frozen prospective evidence and a small number of adversarial, predeclared
tests—not another broad parameter search.
