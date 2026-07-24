# Trade Bot System Whitepaper

Status: canonical overview. Last reviewed: 2026-07-22.

For an exhaustive machine-oriented audit packet, including exact formulas,
current attribution, provenance boundaries, empirical tables, failure modes,
and questions for an independent LLM reviewer, see
[`ai_review_whitepaper.md`](ai_review_whitepaper.md).

## Executive Summary

Trade Bot is a local, human-reviewed trading research and monitoring system for
long-only swing and momentum strategies. It is designed for users who want a
disciplined way to research tactical allocation, size risk, monitor paper
strategies, and record real or simulated decisions without turning the process
into intraday automation. The system does not place trades automatically. It
generates evidence, suggested targets, risk context, and journal-ready tickets
that a human reviews before any action.

The core design problem is not simply "forecast the market." The more useful
question is: under the current market state, which tested strategy family has a
reasonable chance of preserving upside while avoiding unacceptable left-tail
risk? Trade Bot approaches that problem by separating the system into distinct
layers: data intake, current-state diagnostics, scenario mapping, strategy
construction, backtesting, risk management, outcome simulation, paper monitoring,
and execution journaling. Each layer has its own role and tests, so narrative
context does not automatically become a trade and a historically strong strategy
does not automatically become a live position.

The most promising research pattern so far has been high-growth exposure with
explicit off-ramp and re-entry logic. In practical terms, the best candidates
have not been permanently defensive systems, nor simple buy-and-hold clones.
They have generally kept exposure to strong growth assets while using trend,
breadth, volatility, credit, scenario pressure, drawdown controls, and decision
sanity checks to reduce severe drawdowns and re-enter after stress. The leading
candidate family currently sits in the high-growth, moderate-drawdown area of
the research frontier, but those figures are historical research outputs, not
forecasts. The validation framework explicitly preserves skepticism around
theme concentration, transaction-cost sensitivity, live availability of newer
assets, and whether the historical leadership regime can persist.

Trade Bot is also built around the reality that implementation matters. A
strategy that looks good in a backtest still has to be monitored from a specific
start date, valued forward, translated into target weights, logged as tickets,
and reconciled against the user's actual paper or live book. The system therefore
includes champion/challenger monitoring, paper valuations, book alignment,
recommendation tickets, execution logs, taxable-account estimates, and a
Forward Test area. The goal is to move from "interesting research" to a repeatable
process that can be reviewed, challenged, and audited before capital is scaled.

Strategy comparisons use a separate fail-closed contract. The canonical replay
library records the exact price-frame hash and columns, market window, execution
lag, rebalance cadence, costs, configuration, dependencies, and source tree for
every scorecard. It also freezes the outcome-planning basis, including the
$220,000 starting value, $4,000 annual contribution, monthly cadence, 15-year
horizon, and drawdown bands. Its root manifest is written only after every saved candidate
and configured strategy has been replayed and every declared artifact verifies.
The dashboard does not merge earlier-regime scorecards or allow live-snapshot
rows to override canonical replay rows. An incomplete or stale library produces
no comparative leaderboard rather than a mixed one.

The current rebuild contains 427 rows across 73 experiment groups: 406 exact
saved-candidate replays plus 21 configured strategies. All 406 historical rows
matched one-to-one. Relative to the earlier execution regime, median CAGR moved
by 0.003 percentage points and median maximum drawdown by -0.237 points, but 114
promotion decisions changed and nine names entered/exited the top 20. That is
why the cleanup matters despite small aggregate headline changes. Holding the
new return paths fixed and changing only annual contributions from $70,000 to
$4,000 left the top-20 utility set unchanged; 223 lower ranks changed, largely
inside tied score bands. The financial-assumption error materially overstated
terminal dollars but did not create a new champion.

## 1. The Problem Trade Bot Is Built To Solve

Most individual trading tools fall into one of three categories. Some are market
dashboards that show many indicators but do not provide a testable strategy
workflow. Some are backtesting libraries that can evaluate rules but do not
translate them into an operating process. Others are brokerage or portfolio
trackers that record holdings but do not explain why a target changed. Trade Bot
tries to connect those pieces.

The operating constraint is important: this is not a high-frequency or intraday
system. The expected cadence is human-executable, with trades considered a few
times per week at most and often less frequently. The strategy universe is
long-only stocks and ETFs, with no default shorting, leverage, options, or
automatic broker execution. This framing changes the math. Latency still
matters, but not at the millisecond level. Drawdown control matters, but the
system cannot depend on a perfect stop-loss fill. Re-entry matters because a
system that exits risk but never gets back in can underperform badly in recovery
regimes.

The system therefore treats trading as a sequence of research and operating
questions:

- What is the current market state?
- Which signals are proven drivers versus only context?
- Which strategies have worked across regimes, not only one narrow period?
- How much risk is acceptable for a 15-year accumulation problem?
- Is today's recommendation materially different from the current book?
- If the system is paper monitored, is it behaving like its backtest?
- If a trade is made, what was the exact time, price, size, and rationale?

The answer to those questions is not one model. It is a workflow where models,
rules, diagnostics, and human review each have a controlled role.

## 2. System Architecture

Trade Bot is organized as a local Python application with a Streamlit dashboard,
Poetry-managed environment, local cached data, and DuckDB-backed storage. The
dashboard is intentionally not the only system. Most heavy work is designed to
run as batch or CLI jobs, then the dashboard reads precomputed snapshots so the
interactive experience stays responsive.

Dashboard V2 is now the primary operating surface. It is summary-first and
snapshot-backed: routine pages read the latest saved snapshot, DuckDB warehouse
tables, and persisted research artifacts before loading expensive diagnostics.
The archived V1 dashboard remains available only for comparison/debugging. This
keeps the daily workflow fast while preserving access to the full workbench
when the user intentionally opens deeper diagnostic views.

At a high level:

```text
Data and configs
  -> daily snapshot
  -> current-state and scenario engine
  -> risk and sizing engine
  -> operating brief and book alignment
  -> human review, tickets, paper/live journal

Research candidates
  -> backtests and validation
  -> outcome utility and curation
  -> forward simulation
  -> champion/challenger monitoring
  -> promotion, demotion, or archive
```

The main components are:

| Component | Purpose |
| --- | --- |
| Data intake | Loads market prices, macro series, curated events, and news/context into local caches. |
| Current-state engine | Builds the daily market read: risk status, confirmation matrix, 0M regime pulse/instability nowcasts, scenarios, drivers, and current posture. |
| Strategy engine | Constructs target weights for baselines, research candidates, and selected operating systems. |
| Backtest engine | Applies execution lag, rebalance cadence, transaction costs, target weights, and equity compounding. |
| Risk engine | Applies price-state sizing, calibration-gated scenario authority, expected shortfall, stress loss, factor exposure, concentration checks, and defensive floors. |
| Research Lab | Compares experiments, strategy families, outcome frontier, validation, factor attribution, and candidate details. |
| Simulation Lab | Projects selected strategies through deterministic, bootstrap, and regime-conditioned forward paths, then validates those simulations with rolling-origin calibration tests. |
| Launch Lab | Tests whether new or scale-up capital should enter a selected strategy now, gradually, or wait, with Simulation Lab diagnostics acting as a forward-risk guardrail. |
| Experiment Operator | Converts a selected candidate into a paper/live trial contract with suggested horizon, trial capital, launch path, checkpoint criteria, and validate/continue/fail language. |
| Monitoring | Tracks champion/challenger/reference windows from chosen start dates using paper valuations. |
| Forward Test | Records locked recommendations, paper/live executions, current book alignment, and allocation history. |

The separation is intentional. News can explain a concern without driving a
trade. A strong backtest can be promoted for monitoring without being trusted
with capital. A daily action can be small even when a research strategy has a
high long-term score, because the current book may already be aligned.

## 3. Data, Signals, And Scenario Mapping

Trade Bot ingests several classes of data. The most reliable layer is market
price data: ETFs, benchmark indices, sector proxies, factor proxies, and selected
single names. The macro layer includes FRED-style series for rates, credit,
inflation, labor, liquidity, financial conditions, commodities, and related
groups. The news/event layer is more deliberately constrained. It can highlight
AI capex, private credit, IPO/equity supply, policy risk, energy shocks, and
sector-specific narratives, but these items are not automatically treated as
trade drivers unless they map to tested signals or market confirmation.

The current-state engine turns these inputs into a daily risk posture. The
important objects include:

- **Risk status**: a compact label such as green, yellow, orange, or red that
  summarizes current market pressure.
- **Risk score**: a numeric pressure summary behind the status. Higher means
  more risk/defense pressure; lower is more constructive.
- **Confirmation matrix**: a collection of market, credit, breadth, trend,
  volatility, macro, and related checks that can be bullish, neutral, or bearish.
- **Scenario lattice**: probabilities for possible future market regimes across
  horizons such as 1 week, 1 month, 3 months, and 6 months.
- **Driver rotation**: a view that separates historically proven drivers,
  currently active drivers, emerging drivers, fading drivers, and explainer-only
  context.

The scenario layer does not claim to predict exact index levels. It maps current
conditions into broad future-state probabilities. Examples include risk-off,
transition, broad risk-on, fragile risk-on, or risk-off-then-relief. As of the
July 2026 calibration, those probabilities are research context rather than an
allocation driver: their one-month Brier skill was negative and their earned
one-month sizing authority was zero. The raw probabilities remain visible, but
they do not adjust risk budgets or portfolio limits until walk-forward
calibration earns authority through an explicit configuration change.

One important design principle is model authority. Signals are not all equal.
Trade Bot distinguishes:

- **Allocation drivers**: tested inputs that can affect sizing or selection.
- **Validated context**: useful contextual signals with some empirical support,
  but limited authority.
- **Explainer-only context**: narrative or diagnostic items used to explain the
  market, not to move the portfolio directly.
- **Unsupported watchlist items**: things worth watching but not allowed to
  influence the action layer until tested.

This prevents the system from drifting into "trade by vibes." A compelling
headline can enter the daily brief, but a large de-risking move should generally
require confirmation from price, breadth, volatility, credit, trend, or another
validated driver.

The operating decision now stores a sequential causal attribution. It begins
with the base strategy, then records the marginal defensive percentage points
from calibration-gated quantitative risk timing, scenario probabilities, event/news pressure,
accepted macro signals, independent portfolio constraints, and the final
governance guardrail. A layer with zero marginal effect is not described as a
reason for the target. Permanent counterfactuals rerun the decision with news
disabled, news visible but informational-only, and news sizing enabled as a
research comparison.

After the execution and timing repairs, the July 21 cached-data snapshot starts
and finishes at 60.02% native defense. Yellow price fragility is a diagnostic
and adds zero because risk-timing authority is zero. Revised-history macro data
also has zero sizing authority; scenario probabilities, news/events, portfolio
hard constraints, and decision sanity add zero. The news-disabled and
news-informational-only counterfactuals reproduced the same risk score, risk
budget, scenario probabilities, weights, equity beta, and beta-adjusted S&P
delta. This is direct evidence that the current recommendation is not the
research narrative being reflected back through sizing.

## 4. Strategy Construction And Risk Management

Trade Bot strategy candidates are built as long-only allocation systems. Some
are simple references, such as buy-and-hold SPY, buy-and-hold QQQ, BIL/cash,
or a 60/40-style portfolio. Others are tactical systems that rotate between
risk assets and defensive assets. More complex candidates include AI/growth
exposure, sector/factor rotation, broad market re-entry, multi-asset defense,
decision-sanity overlays, volatility targets, and ML or Bayesian probability
guards.

The system separates three concepts that are often mixed together:

1. **Selection**: what assets are eligible for the strategy.
2. **Sizing**: how much of the portfolio should be in each sleeve.
3. **Risk control**: when exposure is reduced, capped, re-entered, or delayed.

For example, a growth-oriented strategy may select QQQ, SMH, SPY, IWM, or
related growth proxies. Its sizing logic may use momentum, drawdown repair,
or volatility targeting. Scenario pressure can regain sizing authority only
after calibration. Its risk control may impose
defensive allocations to BIL, Treasuries, gold, or cash-like assets when
confirmation breaks.

The risk engine sits after the initial strategy target. It checks whether the
raw target violates broader portfolio constraints. It can adjust for:

- factor exposure,
- equity beta,
- AI/growth beta,
- rates/duration exposure,
- credit sensitivity,
- commodity exposure,
- expected shortfall,
- stress loss,
- concentration,
- scenario-weighted downside risk as an advisory diagnostic unless separately
  granted authority,
- minimum defensive allocation.

The balanced-asymmetric operating profile keeps news and narrative
informational-only. A separate decision-sanity overlay remains as a governance
backstop for research configurations that intentionally grant event sizing
authority: it caps event-only defensive moves unless enough market-confirmed
breaks appear.

Risk and narrative semantics are now explicit. The configured defensive asset
is exempt from the risk-asset single-position cap, constraint comparisons use
small absolute and relative tolerances so machine-precision noise cannot create
a breach, and the headline summary is built from the same complete hard-
constraint set as the detailed report. News cache fallback is marked with its
original data-as-of time and is triage-only: degraded source health cannot
create a sizing-authority event. Curated and news-derived events can decay and
expire, so old narrative pressure cannot remain at full strength indefinitely.

Portfolio risk is separated into independently measurable hard constraints and
scenario-conditioned research. Equity beta, expected shortfall, and maximum
stress loss remain hard under the active utility profile. Scenario-conditioned
tightening and the scenario-weighted-stress clamp currently have zero authority,
which removes the structural tendency to converge near 77% BIL while preserving
true catastrophic-risk controls. Utility profiles make the distinction explicit:
growth, balanced-asymmetric, and capital-preservation profiles carry different
normal-tail and catastrophic-stress tolerances; they do not override a failed
scenario-calibration gate.

## 5. What The Research Has Found About Strong Strategies

The most useful pattern so far is not "always own QQQ" and not "hide in cash."
The best historical candidates have generally combined high-growth participation
with drawdown-sensitive off-ramps and re-entry logic. The tradeoff that matters
for the intended use case is not maximum smoothness. It is high terminal wealth
with drawdowns that are painful but survivable.

Current research artifacts show several high-ranking candidates near the
growth-constrained frontier. The daily operating family is centered on
re-entry, volatility targeting, trend repair, and drawdown guards rather than a
single static allocation. The operating choice should not be treated as simply
the highest CAGR row. It is a compromise among historical return, drawdown,
turnover, human-operable cadence, current launch evidence, and validation
quality.

Those are still backtested and validated research results, not live performance
promises. The QC gauntlet is designed to keep the strong result from being
over-read. It checks whether the result survives future-price perturbation,
added execution lag, transaction-cost stress, rebalance-day changes, and removal
of important assets or themes. The leadership diagnostics separately measure how
much of the result depends on technology, AI, semiconductor, or mega-cap growth
leadership. These warnings do not invalidate the leading candidate family, but
they mean the result should be treated as a promising growth strategy with
concentration risk, not as a generic market timing law.

The latest execution work makes that caveat concrete: ordinary rebalance-day
and signal-lag changes materially alter the i111 path. Neither the V2.2 native
hardening mechanisms nor the V2.3 fixed-slate smoothing transforms cleared
their replacement gates, so they remain research evidence rather than operating
strategy upgrades.

A subsequent fixed cross-sectional study tested one causal hypothesis rather
than another parameter grid. When six of the native strategy's eight AI-stress
components agreed, AI exits stayed immediate but new or increased AI targets
were redirected to either BIL or RSP. Neither policy passed. At configured
costs, the reference produced 21.01% Wednesday CAGR and a -30.60% worst
execution-profile drawdown. BIL deferral reduced Wednesday CAGR to 18.85%,
barely improved the worst drawdown to -30.32%, and increased execution failures
from seven to eight. RSP deferral produced 19.29% Wednesday CAGR, worsened the
worst drawdown to -34.29%, and also had eight failures. This closes that fixed
hypothesis as `no_robust_improvement`; it does not justify a strategy change.

A later native selector/transition study tested incumbent buffers, 63/126-day
blended ranks, a 15% SPY/RSP core, recovery-metered entry, and their
combinations without adding another defensive overlay. None cleared the fixed
replacement screen. The incumbent buffer modestly raised configured CAGR from
20.79% to 20.91% and median execution CAGR from 19.54% to 19.80%, but worsened
configured drawdown by 0.69 points and improved only 40.9% of calendar years.
A post-initial blended-rank-plus-buffer ablation reduced configured CAGR to
20.36% while improving median execution CAGR to 20.13%, worst execution
drawdown by 1.01 points, and the Aug. 2023-Jan. 2024 result by 3.20 points.
That is useful evidence about turnover and transition fragility, not a validated
replacement. The core and recovery-meter mechanisms were return-dilutive, so
the operating strategy remains unchanged.

Later sector-regime and global-rotation experiments serve a different purpose.
They have produced lower-CAGR candidates, often in the 3-6% range, with lower
drawdowns and useful macro-rotation diagnostics. They are not currently
displacing the i111 high-growth engine as the daily operating candidate. Their
main value is as context, stress comparison, and future diversification research.

The recurring traits of stronger candidates are:

- They preserve meaningful exposure to growth when trend and breadth are not
  broken.
- They reduce exposure when drawdown, volatility, credit, or confirmation
  pressure becomes meaningful.
- They re-enter risk faster after repair signals improve.
- They do not let a single narrative headline force a huge defensive move.
- They compare well on terminal-wealth utility, not only Calmar or Sharpe.
- They remain practical enough for human trading cadence.

Some strategy families looked intuitively attractive but were less useful in
testing. Very defensive systems often reduced drawdowns but produced insufficient
CAGR for the accumulation objective. Some source-of-funds and broad rotation
ideas added useful context but gave up too much return when used as the core
system. Several ML routers reduced risk but created overly conservative behavior.
The current best use of those weaker components is usually diagnostic, gating,
or satellite context, not replacement of the growth/re-entry engine.

Factor attribution has also been useful. Many strategies that look different on
paper are partly the same bet: growth beta plus defensive timing. That does not
make them bad, but it does mean the system should not monitor ten versions of
the same exposure and pretend they are independent challengers. The dashboard
therefore includes family maps, factor attribution, residual behavior, and
curated shelves to make strategy overlap visible.

### Broad Defensive-Bias Calibration

The July 23 defensive-bias study tested whether Trade Bot is systematically too
defensive too early across every eligible month-end, not only bubble peaks or
large drawdowns. It covered 11 dynamic risk-managed strategies after a
three-year warm-up and produced 3,172 one- and three-month frozen-weight
counterfactual rows. Static allocations were excluded. Near-clone strategies
could not multiply the global or family sample because pooled evidence was
averaged by origin.

The candidate adjustment was deliberately small: at most five percentage points
between BIL/residual defense and the strategy's existing risk sleeve. It could
not invent a risky holding, defense relief was blocked during confirmed or
severe market breaks, and hard portfolio-risk constraints were excluded because
they encode loss tolerance rather than a forecast error. Every online estimate
used only outcomes whose forward window had already matured.

The broad result is useful but not strong enough to change the live policy:

- Ordinary-market defense-relief cases were positive on the declared one-month
  utility in 68.0% of rows across 23 unique origins, but only six strategies
  produced such cases. Named-stress relief was positive in 51.1% of rows across
  36 origins.
- Risk restraint was not the mirror image. In ordinary markets it was positive
  in 53.7% of one-month rows, but average return delta was negative and the
  three-month utility edge disappeared.
- The best naive fixed symmetric rule raised median strategy CAGR by 0.16
  percentage points and had non-worse drawdown in 63.6% of strategies, but only
  half of calendar eras had a positive median return effect and only 71.6% of
  crisis tests had non-worse drawdown. It failed the predeclared gate.
- The hierarchical defense-relief candidate raised the focus strategy CAGR by
  only 0.03 points while worsening maximum drawdown by 0.61 points. Only 45.5%
  of strategies and 25% of eras had positive return deltas.

No pre-registered candidate passed the retrospective gate, and prospective
evidence is absent. Allocation authority therefore remains zero. Today's 57.68%
defensive focus-strategy weight is below the 60% research trigger, so the
calibrator proposes no current adjustment. The most defensible conclusion is
not “always add risk because Trade Bot is early.” It is narrower: early defense
contains some strategy-family-specific upside-regret information, especially in
recent i111 history, but the bias is not stable enough across eras and
drawdowns to estimate as a live correction.

A subsequent architecture search went beyond threshold tuning and tested 30
distinct mechanisms: trend, credit, volatility, breadth, momentum, drawdown,
defense duration, ramp speed, recovery, risk floors, family disagreement,
opportunity-cost feedback, re-entry acceleration, SPY/SPLV/RSP bridge sleeves,
and conditional combinations. None cleared all eight fixed gates.

The result is not equivalent to “nothing helps.” Breadth-gated five-point relief
was a real near-miss. For the focus strategy it added 0.30 CAGR points, cost 0.41
maximum-drawdown points, acted on 6.86% of days, remained positive at 10 and 20
basis-point costs, had positive cross-strategy median return effects in all four
eras, and passed the crisis gate. It cleared seven of eight gates. It failed
because only 54.5% of strategies had non-worse maximum drawdown versus the 60%
requirement. The damage was concentrated in i111 variants; unrelated dynamic
strategies were mostly unchanged.

This was not a crisis-only test. The common history contains 5,422 daily
sessions from 2005-01-03 through 2026-07-23. An explicit diagnostic removes all
eight named crisis windows, leaving 3,592 ordinary-market sessions.
`breadth_intact_relief` adds 0.42 annualized return points to the focus path in
that crisis-excluded sample, and all 11 strategies have non-worse ordinary-path
maximum drawdown, but only 54.5% have higher ordinary-path annualized return.
Across quarterly-sampled rolling windows, the focus rule beats base on return in
54.2% of one-year and 73.3% of three-year windows. That confirms a real
opportunity-cost effect outside crises, but not a uniformly reliable one; its
full-path drawdown damage arises in stress windows.

Second-wave breadth combinations could reduce the drawdown cost, but only by
becoming too rare to satisfy the material-allocation gate. Two mechanisms would
change today's allocation: native re-entry acceleration would reduce defense by
five points and an intact-trend risk floor by 2.68 points. Both failed multiple
historical gates, so neither has authority. The current conclusion is therefore
more informative than the first study: a small breadth-conditioned relief sleeve
is the leading prospective research candidate, but the system cannot honestly
claim a robust live correction yet.

The next study removed the 60% trigger entirely. It fit nested,
episode-weighted break-progression models and translated probabilities into a
continuous defensive target. The hazard forecast did not validate: its mean
Brier score was worse than the expanding break base rate, the result was not
stable at 8%, 10%, and 12% break labels, and most leave-crisis-cluster-out tests
failed. Confirmation acceleration and warning-age decay also reduced return.
This rejects a broad hazard-driven rebuild.

However, the experiment exposed the earlier search constraint. Every initial
fold selected the mildest curve boundary, so a clearly post-hoc lower-slope
extension tested continuous i111 defense calibration. It raised focus CAGR from
29.86% to 32.89% within the 2015-2026 nested outer-test window while worsening
that window's maximum drawdown from -25.80% to -26.37%. These are not comparable
with the roughly 20.67% full-history 2005-2026 configured-path CAGR; the valid
same-window improvement is 3.03 points. All four folds improved CAGR, but two
fold drawdowns worsened by 2.61 and 4.35 points. All six i111 variants improved,
while the broader
dynamic-risk-managed family did not.

The effect survives 20-basis-point costs and one- or two-session extra execution
lags. In 1,000 paired 63-session block resamples, the CAGR delta was positive in
all samples, but 27% worsened drawdown by more than one point. Today the curve
would reduce focus defense from 57.68% to 47.92%. This is a meaningful,
explicit return-versus-drawdown trade, not a free correction. Because the
lower-slope grid was motivated by observed boundary behavior, it is frozen only
as `i111_continuous_defense_calibration_v1` for prospective shadow monitoring
with zero allocation authority.

Candidate Details now places a drawdown-attribution diagnostic directly below
the combined performance/drawdown/allocation chart. For the selected history
window it reports the local peak, trough, recovery measurement, gross loss
contributors, peak-to-trough turnover and costs, risk/defensive exposure path,
and missed SPY/QQQ recovery. Asset contributions are arithmetic gross
diagnostics and are explicitly not presented as an exact compounded
reconciliation.

## 6. Testing And Validation Framework

Trade Bot uses several layers of testing because no single backtest statistic is
trustworthy enough on its own.

The first layer is the basic historical backtest. It applies target weights,
signal lag, rebalance cadence, transaction costs, and portfolio compounding.
This produces returns, CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar,
turnover, Ulcer Index, allocation history, and transaction behavior.

The operating close-only convention now uses a two-row shift. A target using
close `t` is modeled as filled at close `t+1` and begins earning the following
close-to-close interval. The former one-row shift first labeled the position on
`t+1` but earned the `t`-to-`t+1` return, which is an implicit boundary fill at
the same close used by the signal. It remains visible only as a labeled research
approximation. Under the clean lag-2 path, the configured primary produced
20.67% CAGR / -25.80% maximum drawdown, and the native i111 challenger produced
20.84% / -24.75%. The old 22.18% / -19.68% Wednesday figure is not an operating
headline.

The second layer is rolling-window and regime testing. The system evaluates
performance across shorter windows such as 1 year, 3 years, and 5 years, as well
as calendar-year and market-regime slices. This matters because a strategy that
only works in one historical era is not enough. The user can inspect periods
such as 2008, 2011, 2018, 2020, 2022, and recent AI-led markets to understand
where a strategy exited, stayed defensive, re-entered, or failed.

The third layer separates two tests that should not be conflated. The common
`walk_forward.csv` artifacts are sequential fixed-strategy holdouts: the same
already-defined strategy is evaluated across later one-year windows. They show
time-local fragility, but they do not train, tune, or select on the preceding
segment and therefore are not proof against selection overfit. True
walk-forward selectors and routers use only an earlier segment to choose a
model or rule, then score that frozen choice on a later segment. A strong
candidate should have positive sequential-holdout behavior, tolerable worst
rolling periods, reasonable left-tail performance, and—where selection is part
of the mechanism—a genuinely nested train/select/test record.

The fourth layer is outcome utility. For this project, the objective is not just
"high Calmar." The Growth-Constrained Outcome Frontier asks whether extra CAGR
is worth additional drawdown for a 15-year accumulation problem with ongoing
monthly contributions. It estimates terminal wealth, benchmark-relative wealth,
recovery return required after drawdown, Ulcer Index, drawdown penalties, and
validation confidence. This makes it possible to compare an 11% CAGR, -15%
drawdown system against a 22% CAGR, -20% drawdown system in terms of actual
wealth utility and recovery burden.

The fifth layer is ablation and signal evidence. The system can ask whether a
signal family improves CAGR, drawdown, re-entry, churn, or left-tail behavior
after costs. This is how narrative or newly added indicators should earn their
way into the action layer. If a signal cannot be tested or mapped to a relevant
now-casting framework, it should remain explanatory context.

The sixth layer is the backtest-QC gauntlet. This is a structural-skepticism
audit for the leading candidate, not another performance chart. It perturbs
future prices after a cutoff to detect look-ahead leakage, checks extra signal
lag, stresses transaction costs, changes rebalance day, removes key assets or
themes, and reports contributor concentration. A candidate that performs well
but collapses under these stresses should remain research-only until the
fragility is understood.

The seventh layer is the probability-of-backtest-overfitting gauntlet. This is
the multiple-comparisons audit. The system builds a synchronized return matrix
for the selected candidate shelf, partitions history into equal blocks, tests
every symmetric half-history train/test split, and asks whether the
in-sample winner lands above or below the median out-of-sample result. Low PBO
supports the research process. High PBO means the strongest-looking backtests
may be artifacts of trying many variants on the same market history.

The July 22 canonical 20-candidate shelf produced 64.29% PBO, labeled
`high_overfit_risk`, with 0% OOS-loss probability. The narrower active
12-candidate adversarial roster produced 21.43%. That disagreement is itself a
warning: PBO is roster-dependent and should not be quoted without its shelf.

The eighth layer is defensive-signal judgement. This audits moments when the
strategy moved heavily into BIL or residual cash and asks whether that caution
was historically useful, a false alarm, or mixed. It reports the episode count,
correct-defense rate, false-alarm rate, missed upside, avoided drawdown,
benchmark comparison, and re-risk behavior by horizon. The point is not to turn
this into a separate sizing engine. It is an interpretability layer that says,
in plain language, whether today's defensive posture resembles historical
caution windows that helped or mostly missed upside.

The calibration is also applied to the scenario probabilities themselves.
Point-in-time weekly origins are matched to matured SPY-versus-BIL and drawdown
outcomes at one week, one month, and three months. The report includes reliability
bins, Brier score and skill, AUC, expected calibration error, block-bootstrap
uncertainty, and expanding-history authority. In the July 22, 2026 run, one-week
and one-month Brier skill were negative; three-month skill was slightly positive
but its interval crossed zero. The active policy therefore assigns zero
one-month sizing authority.

The layered-defense audit found that the old intersection of base defense,
scenario clamp, and portfolio stress clamp did not add reliable downside
discrimination. At one month it had 27 independent episodes, a 22% correct-
defense rate, and a 52% costly-false-positive rate; at three months it had 26
episodes, a 27% correct-defense rate, and a 50% costly-false-positive rate.
The old final defensive weight also clustered near 77%, confirming a structural
attractor rather than a uniquely precise current estimate. These retrospective,
small-sample findings justified removing authority.

The replacement-policy replay then retested 1,020 weekly origins with the live
calibration gates applied. Scenario probabilities added exactly zero defense at
every origin, as required by the zero-authority configuration. At the study's
materiality thresholds, the current state is not a three-layer agreement: the
native strategy is defensive, price-derived timing adds zero operating defense,
and hard portfolio
constraints add zero. No historical origin had native defense above 55%, a
quantitative sizing addition above five points, and an additional hard-
portfolio clamp above one point simultaneously.

The closest research comparison, native defense plus confirmation-timed sizing,
is directionally interesting but far too small for authority. At one month it
produced 10 episode starts, a 40% correct-defense rate, and a 20% false-alarm
rate, versus 48 native-only starts at 38% and 44%. At three months its rates
were 50% and 30%, versus 34% and 38%. By contrast, quantitative timing plus a
portfolio clamp without native defense had weaker one-month discrimination than
timing alone: 41% beneficial-under-rule and 39% costly-false-positive across 44
starts, versus 53% and 26% across 19. The non-random cohorts and small confirmed-timing sample
do not establish an optimal threshold.

The refreshed non-overlapping weekly replay makes the opportunity cost explicit.
Native sizing produced a 19.60% CAGR and -25.90% maximum drawdown. Legacy
risk-status sizing improved drawdown by 4.80 percentage points but reduced CAGR
by 5.20 points. The confirmation-timed candidate reduced CAGR by 0.50 points and
made maximum drawdown 0.40 points worse; adding hard portfolio limits improved
drawdown by 5.10 points but reduced CAGR by 5.10 points. The current authorized
overlay cost 0.10 CAGR points and made drawdown 0.30 points worse. The candidate therefore
remains visible with zero allocation authority. A separate sparse pre-break
overlay replay reduced median CAGR from 13.09% to 11.94% without improving the
-22.10% median maximum drawdown. These are
retrospective, current-universe results rather than prospective proof, but they
show that the cost of optionality has not historically been small enough to
treat aggressive overlay defense as a free improvement.

The pre-break population is no longer an event-heavy panel paired with only
recent controls. It now contains 572 point-in-time origins: 397 weekly origins
around eight named breaks and 175 monthly controls outside those windows from
2005 through 2026. The 571 mature outcomes reduce to 71 conservative event or
calendar-quarter clusters. Cluster-bootstrap intervals exclude zero for only
12 of 128 signal associations. Cross-sectional dispersion is the strongest
association (Spearman 0.34, 95% interval 0.11 to 0.51), but this remains a
hindsight monitor and has no independent allocation authority.

The 1:1 historical snapshot replacement also changed the causal diagnosis of
early hard defense. Quantitative timing now adds zero. Absolute portfolio-risk
constraints account for 81.8% of Early Watch hard-defense snapshots and 69.5%
of Long Lead hard-defense snapshots; the native strategy accounts for most of
the remainder. Those constraints are utility/risk-tolerance controls rather
than forecasts, so they remain intact here. Their calibration is the next
separate risk-engine question.

A narrower replay that applied the final overlay only during the current
material layer classification (`base_only` at the 55%/5%/1% thresholds) was
nearly neutral: it reduced CAGR by 0.06 percentage points and improved maximum
drawdown by 0.27 points. That trade-off is too small and retrospective to
justify sizing authority. Under the repaired policy,
the timing candidate is research-only rather than a modest discretionary
adjustment. This prevents its warning state from being treated as independent
confirmation of a high-conviction break call.

The separate strategy-native audit reaches a similarly measured conclusion. At
the focus strategy's 65% defensive threshold, one-month episodes were 45.2%
beneficial under the stated rule, 33.3% costly false positives, and 21.4% mixed.
At three months they were 42.9%, 31.0%, and 26.2%. Native defense has useful but imperfect historical
discrimination; it should not be described as a crash prediction.

The contribution-aware catastrophic-tail read uses 1,000 fixed-seed,
21-session block-bootstrap paths over 15 years, starting at $220,000 with
$4,000 annual contributions. Drawdown is calculated on a flow-neutral return
index so contributions cannot hide losses. For the configured primary, the
historical resample produced a 48.5% frequency of drawdown beyond 25%, a 22.1%
frequency beyond 30%, and a 98.5% chance of exceeding the wealth generated by a
deterministic 10% return path. These are modern-universe historical resamples,
not forecasts; their value is comparing policy utility under one frozen method.

The ninth layer is leadership-dependence diagnostics. This exists because the
strongest candidates can be excellent for reasons that are too concentrated.
The diagnostic report measures each top candidate's historical and current
exposure to QQQ, SMH, SOXX, IGV, and single-name mega-cap technology, its beta
to QQQ, SMH, SOXX, SPY, VEA, IWM, GLD, and TLT, its return contribution by
asset, its behavior when QQQ underperformed SPY, RSP, or VEA, and its
performance across scenario buckets. It also runs leadership-impairment stresses
that haircut technology returns or reallocate technology exposure toward global
breadth, real assets, and defensive alternatives. The purpose is to separate a
durable risk-managed growth process from a post-hoc expression of one dominant
leadership theme.

The same report includes a walk-forward strategy router. At each historical
origin, the router uses only prior observations for the candidate set, scores
which strategies performed best in similar prior scenario states, and then
evaluates the selected or blended strategy over the next 1, 3, and 6 months.
This is not a full research-process out-of-sample proof, because the candidate
menu itself is known today. It is still useful because it answers a narrower
operational question: given today's strategy shelf, would a state-aware,
prior-only router have preferred a better candidate or blend when the world
looked similar? The benefit is not automatic strategy switching. The benefit is
knowing whether current scenario context historically improved candidate
preference, whether the advantage appears only at longer horizons, and whether a
blend is safer than a single winner.

The tenth layer is the Scenario / Phase Frontier, also called the Speculative
Cycle Tracker. This layer addresses a narrower question raised by the current
AI/growth leadership debate: if the market is in a speculative cycle, what phase
does it most resemble now, what phases are plausible over 1-month to 1-year
horizons, and which assets historically behaved better after similar prior-only
phase reads? The phase taxonomy includes normal cycle, acceleration, pre-break,
early unwind, liquidation, bottoming, recovery, and post-unwind compounding.
This is not a crash timer and it is not an allocation override. It is a
research/watch layer that organizes evidence about phase risk and possible
post-unwind winners.

The Cycle Tracker has strict leakage controls. Each historical origin rebuilds
phase features only from prices available through that date, then evaluates
forward returns starting on the next trading session. Current scenario
probabilities can shape the current horizon frontier, but historical validation
does not apply today's scenario state to past origins. The module writes
artifact CSVs and DuckDB tables for phase probabilities, horizon phase
frontiers, evidence components, current-phase conditional candidate scores,
phase-by-horizon winner shelves, and prior-only validation metrics. The winner
shelf is deliberately conditional: it lets a user inspect candidates for
acceleration, pre-break, unwind, liquidation, bottoming, recovery, or
post-unwind compounding instead of pretending that one deterministic future
state is known today. V2 Research reads those persisted outputs rather than
running the expensive validation in the dashboard.

The Cycle Tracker deliberately separates two trust checks. Path Reliability
audits the path-constrained operational read after sequence rules, prior phase
memory, duration, and drawdown preconditions are applied. It asks whether
historical origins with that path-aware label behaved as the label implied over
the selected horizon. Historical Phase Reliability audits the raw evidence
label before those path constraints. This split is important because raw
evidence can look acceleration-like while the path-aware state says the market
is better framed as post-unwind compounding. The operational read is the
path-aware one; the raw evidence remains visible so disagreements can be
inspected instead of hidden.

Research governance is also a test layer. New manifests record the declared
trial roster and run a fail-closed point-in-time universe audit covering
historical membership, holding-date eligibility, delisting returns, and source
metadata. The consolidated 2026-07-22 ledger indexes 569 manifested completed
trial rows across 19 manifests and 16 distinct studies. Five manifests lack an
explicit candidate roster, 113 artifact directories have no manifest, and all
569 trial rows still lack verified point-in-time universe evidence. That is a
promotion blocker, not a warning label. The ledger cannot reconstruct
interrupted or unmanifested attempts, so complete historical trial-count proof
also remains unfinished and retrospective promotion is disabled.

Simulation validation is now treated as its own test family rather than a visual
nice-to-have. The rolling-origin simulation test chooses historical origin dates,
trains the simulator only on returns available through each origin, simulates
forward paths for configured horizons, and compares the simulated distribution
with the realized future return and drawdown. The primary calibration question is
whether realized outcomes land inside the simulated interval at approximately
the target rate. For example, a 20th-to-80th percentile band should contain
realized outcomes roughly 60% of the time over enough origins. The test also
records median forecast error, severe-drawdown probability error, and a
launch-action score that translates simulated and realized outcomes into a
simple wait, ramp-in, or full-launch scale.

The simulation validation layer is deliberately more demanding than "did the
chart look plausible?" It stores run-level, horizon-level, origin-level, and
ablation metrics in the local DuckDB warehouse. That makes Simulation Lab a
persistent calibration monitor instead of a one-off dashboard calculation.
Ablations compare the baseline regime sampler, duration-aware transitions,
duration plus covariate-matched blocks, and factor-proxy paths where factor data
is available. If the more complex model does not beat the simpler baseline on
coverage, median error, severe-drawdown calibration, or launch-action quality,
the system should not give the extra complexity more decision authority.

Stored simulation validation runs should be read as calibration evidence rather
than as predictions. A good run can support planning confidence when interval
coverage, median miss, severe-drawdown calibration, and launch-action quality
are all acceptable. A weak run should add friction to Launch Lab and reduce the
authority of the forward fan chart. Simulation Lab is planning support, not a
standalone launch authority.

Rolling-origin validation is checkpointed and resumable. Checkpoints include a
fingerprint of the exact returns, scenario/factor inputs, and validation
settings; a changed-input resume fails closed. A bounded native-i111 smoke test
verified the mechanism across 77 quarterly one-month origins with only 20 paths
per origin. It produced 45.45% interval coverage against a 60% target, 4.57%
median absolute error, 3.90% launch-decision accuracy, and a 44.16% launch-action
score. The distribution label was research-usable, but the action layer was
`action_checks_not_ready`. Because the run used a tiny path count, one horizon,
quarterly origins, and fallback scenario probabilities, it is implementation
evidence and a warning—not full native-i111 simulation validation.

Scenario history is also separated by source. Saved snapshots are true
date-stamped operating records. Reconstructed scenario history rebuilds the
price-derived scenario engine from prices truncated through historical origins,
which is point-in-time safe for the price-driven parts of the current engine.
It is not the same as a full vintage macro or news database. This distinction
matters: reconstructed histories are useful for testing whether the current
price-based scenario logic would have classified past markets sensibly, but they
should not be read as proof that today's macro or narrative layer existed in
that historical form.

Finally, automated software tests protect the math and workflows. Unit and
regression tests cover formulas, dashboard explainability, launch readiness,
forward simulation, outcome utility, paper valuation, and experiment machinery.
This does not prove a strategy will work in the future, but it helps prevent
code drift from silently changing definitions or breaking the operating process.

## 7. Simulation Lab And Forward Planning

The Simulation Lab is the forward-looking planning layer. It exists because a
single historical CAGR number hides path risk. Two strategies with similar CAGR
can feel very different if one has long shallow drawdowns and the other has rare
violent losses. A retirement accumulation problem also depends on ongoing
contributions, not just one starting dollar.

Trade Bot uses several simulation modes:

- **Deterministic accumulation**: applies a historical CAGR to starting capital
  and monthly contributions. This is fast and useful for comparison, but it is
  not a forecast.
- **Historical sequence bootstrap**: resamples realized strategy return blocks
  to estimate terminal-wealth ranges and drawdown distributions.
- **Regime-conditioned forward paths**: samples future paths using current
  scenario probabilities and historical regime-labeled return behavior.
- **Validated rolling-origin paths**: repeatedly reruns the simulator from
  historical origin dates and checks whether simulated ranges, medians, drawdown
  probabilities, and launch-action labels match what actually happened.

The regime-conditioned model is the most distinctive. It is closer to a
scenario-aware Monte Carlo process than a simple CAGR calculator. Each path can
move through a sequence of regimes, such as risk-on, transition, fragile
risk-on, risk-off, or risk-off-then-relief. Strategy behavior is then sampled
from historical windows associated with those regimes. This produces ranges for
terminal wealth, drawdowns, severe-drawdown probability, and benchmark-relative
outcomes.

The current simulator uses three additional controls to reduce the weaknesses of
plain regime resampling:

- **Duration-aware transitions**: the simulator tracks how long the current
  simulated regime has persisted. If a regime is young relative to historical
  median duration, the model modestly increases persistence. If a regime is old
  relative to its historical upper-quartile duration, the model shifts some
  probability toward plausible exit states. This prevents every block boundary
  from acting like a fresh independent regime draw.
- **Covariate-matched return blocks**: historical blocks are sampled from the
  chosen regime, but the sampler can prefer blocks whose starting conditions
  resemble the current state. The default covariates include recent trend,
  medium-term trend, short volatility, drawdown, and short shock behavior, with
  room for external numeric covariates. This preserves non-parametric realized
  return behavior while reducing the chance of sampling a block from a very
  different market state.
- **Factor-proxy paths**: where factor proxies are available, the simulator can
  fit a transparent linear factor model to strategy returns, sample factor
  blocks with the same regime and covariate machinery, and reconstruct strategy
  returns from factor exposure plus residual behavior. This is not a black-box
  alpha model. It is a diagnostic proxy for asking whether the simulated
  strategy path is mostly explained by familiar factor exposures such as equity,
  growth, rates, commodities, credit, or defensive assets.

These controls add realism, but they also add ways to overfit. For that reason
the dashboard and CLI expose ablation results beside the main validation read.
The preferred model should be the simplest variant that gives acceptable
coverage, lower median miss, sensible severe-drawdown calibration, and better
launch-action behavior across the horizons the user actually cares about.

The Simulation Lab also includes interpretability. It should help answer:

- Does the simulated future resemble historical strategy behavior?
- Which regimes dominate the forward path distribution?
- Is the selected strategy outperforming references because of high median
  return, lower severe-drawdown risk, or favorable scenario weighting?
- Where is the model fragile because regime labels are coarse or data is sparse?
- How does the strategy compare to holding SPY or QQQ under the same contribution
  assumptions?

This matters most for the project's 15-year accumulation objective. The simple
Outcome Frontier view answers whether a candidate has enough historical return
and drawdown discipline to deserve attention. Simulation Lab then asks a harder
question: if future paths arrive in different sequences of risk-on, risk-off,
transition, and relief regimes, what range of account values and drawdowns could
the user experience while contributing monthly? That distinction is important
because a strategy can have an attractive deterministic 15-year wealth estimate
and still be unpleasant or unsuitable if simulated paths show deep interim
losses, poor downside percentile outcomes, or too much dependence on one regime.

The simulation approach should be read as a hierarchy rather than one answer.
Deterministic accumulation is the benchmark math. Historical bootstrap adds
sequence risk by reshuffling realized return blocks. Regime-conditioned paths
add current-state awareness by sampling from historical regime-labeled behavior
according to today's scenario map. If all three views tell a similar story, the
planning read is stronger. If deterministic wealth looks excellent but
regime-conditioned paths deteriorate, the strategy may be too dependent on a
favorable historical mix. If the selected strategy beats SPY or QQQ in median
paths but has worse left-tail paths, the user is accepting more dispersion for
potential wealth.

The rolling-origin validation view is how the system decides whether those
simulated ranges deserve trust. A calibrated interval hit rate means the band is
about as wide as advertised, not that the simulator is highly predictive. Median
miss measures whether the central path is useful or merely directional. The
severe-drawdown Brier-style error asks whether simulated drawdown probabilities
behave like probabilities.

The launch-action score is a bridge from simulation quality to operating use.
For each historical origin and horizon, the simulator maps its distribution into
one of three coarse actions: wait, ramp in, or full launch. A non-positive
simulated median or high simulated severe-drawdown probability maps to wait.
A negative lower-band return or elevated severe-drawdown probability maps to
ramp in. Otherwise the simulated action is full launch. The realized future path
is mapped to the same scale using hindsight return and drawdown: negative
return or severe drawdown means wait would have been best, moderate drawdown
means ramp would have been best, and a positive path without meaningful drawdown
means full launch would have been acceptable. Action error is the distance
between those two labels. Over-risk means the simulator was too aggressive
versus hindsight; under-risk means it was too cautious. This is intentionally
blunt: a model that can roughly separate "wait," "ramp," and "full launch"
across historical origins is more useful for operating decisions than one that
only produces attractive looking fan charts.

The simulation layer is planning support. It does not decide trades by itself.
It helps users understand whether a candidate's historical edge is plausible
under future-state assumptions and whether the range of outcomes is acceptable.

## 8. Launch Lab And Entry Timing

Launch Lab answers a different question from both Research Lab and Simulation
Lab. Research Lab asks whether a strategy has worked historically. Simulation
Lab asks what long-horizon future ranges could look like. Launch Lab asks
whether fresh paper or live capital should begin following the strategy now,
phase in over several weeks, or stay on deck.

This distinction exists because adoption timing is a real source of risk. A
strategy can be attractive over 15 years and still have poor short-term entry
windows. Launch Lab therefore tests historical start dates for the selected
strategy using multiple horizons, such as 3 months, 6 months, 1 year, 3 years,
and 5 years. For each start date it compares launch protocols, including an
immediate launch and staged ramps such as 25% now with the remainder phased in
over 4, 8, or 12 weeks. It then evaluates positive-start rate, benchmark beat
rate, bad-start rate, forward return, excess return, max drawdown, and
first-month drawdown.

The key interpretation is horizon-dependent:

- Short windows, especially 3 months and 6 months, are entry-timing stress
  tests. They ask whether starting now has historically been vulnerable to quick
  regret, early drawdown, or underperformance.
- Longer windows, such as 1 year, 3 years, and 5 years, ask whether the strategy
  has had enough time for its dynamic sizing and compounding edge to work after
  a wide variety of historical entry dates.
- If short windows say "wait" or "starter sleeve" but longer windows say
  "phase in," the message is not contradictory. It means the strategy may be
  worth owning for the planning horizon, but new capital should not necessarily
  enter at full size in one trade.

This is especially relevant for dynamically sized strategies. Once a sleeve is
running, the strategy can de-risk or re-risk as its rules change. That makes
entry timing less important than it would be for static buy-and-hold, but it
does not make launch timing irrelevant. New capital still has a first few weeks
or months of exposure, and a full-size launch just before a drawdown can create
behavioral and financial friction even if the strategy later responds correctly.
Launch Lab is therefore a scale-up control, not a daily rebalance engine.

For the 15-year retirement horizon, the practical use is to separate adoption
confidence from tranche timing. One-year, three-year, and five-year launch
evidence should carry more weight when deciding whether a strategy belongs in a
long-term operating set. Three-month and six-month launch evidence should govern
how aggressively to put new dollars to work. A strong long-horizon strategy with
fragile near-term entry evidence may deserve a small starter sleeve or staged
entry rather than a full allocation. A strategy that looks weak across both
short and long horizons should stay out of the operating set.

Launch Lab also exposes whether ramp protocols actually matter. Over a 3-month
or 6-month horizon, a 4-week or 8-week ramp can materially change the first
drawdown experience. Over 3-year or 5-year horizons, ramp choice should matter
less because the ongoing strategy behavior dominates the initial entry
schedule. When all ramp protocols look identical, the user should not over-read
the ramp choice; when they separate meaningfully, staging is adding measurable
entry-risk control.

The aggregate Launch Lab view repeats this question across the curated and
Pareto strategy shelf. It counts how many candidates are wait, set, or ready at
each horizon, shows how labels transition as horizons extend, and reports
whether 4-week, 8-week, and 12-week ramps materially change outcomes. This
matters because launch evidence can be strategy-specific or broad. If many
strong candidates move from wait at 3 months to set or ready at 1 year and
beyond, the system is saying short-term timing is fragile while longer-term
adoption evidence remains constructive. If ramp protocols are effectively
identical, the launch decision should focus on strategy quality and current
risk state rather than over-optimizing the entry schedule.

Like Simulation Lab, Launch Lab is not a forecast. Long windows overlap heavily,
so a large number of 3-year or 5-year historical start tests should not be read
as independent trials. The useful read is comparative: does this strategy have
poor entry behavior across many historical starts, does staging improve that
behavior, and does the answer change when the horizon is aligned to the user's
actual investment problem?

The connection between Launch Lab and Simulation Lab is important but should be
kept explicit. Launch Lab's historical windows answer what happened after prior
entry dates. Simulation Lab answers what the current scenario-conditioned
distribution implies and whether that distribution has been calibrated in
rolling-origin tests. A good launch process needs both. Historical entry windows
can look strong while the forward simulator says the current state has elevated
left-tail risk or weak calibration. Conversely, a simulated forward range can
look acceptable while historical entry windows show repeated bad starts.

The Simulation Gate is the integration point between these two labs. It brings
in horizon-specific validation status, simulated severe-drawdown probability,
median miss, launch-action score, over-risk and under-risk rates, and the active
ablation read. Those diagnostics should not mechanically replace Launch Lab's
historical entry score. Instead, they apply conservative friction: weak
simulation validation caps launch enthusiasm, high simulated left-tail risk
argues for staging, and calibrated constructive simulation evidence can support
a stronger ramp only when historical entry evidence agrees. The most useful
human read is often the disagreement itself: "entry history is constructive, but
simulation confidence is weak," or "simulation looks constructive, but past
launches from similar windows had early drawdown pain."

The Experiment Operator sits beside Launch Lab. Its question is narrower and
more operational: if the user gives Trade Bot a small paper or live cash sleeve
today, how long must that trial run before it proves anything? The operator
builds a trial contract from the selected candidate, current launch gate,
historical signal turnover, and benchmark choice. It recommends a minimum
evidence horizon, a launch path, a trial capital preset, checkpoint language,
and validate/continue/fail criteria. For the high-growth i111 family, the
default benchmark context is QQQ plus BIL/cash, with SPY retained as secondary
broad-market context. The key design principle is that the required horizon is
set upfront, before the experiment starts, so the system cannot keep moving the
goalposts after a lucky or unlucky first week.

## 9. Monitoring, Tickets, And Making It Real

A common failure mode in research systems is stopping at the backtest. Trade Bot
adds the operational plumbing needed to make the research observable.

The Monitoring section creates champion/challenger/reference windows. A champion
is the main strategy currently being followed or most seriously evaluated. A
challenger is a competing candidate. A reference is a benchmark such as SPY, QQQ,
60/40, or cash-like exposure. Each window has a start date, paper capital, daily
valuations, cumulative return, benchmark comparison, drawdown, and forward
status. This is critical because paper monitoring from different start dates can
otherwise create misleading comparisons.

The first genuine no-backfill V2.2 cohort was frozen on 2026-07-21. It contains
the configured primary, the native risk-repair challenger, the Min-25 lower-risk
challenger, QQQ, and SPY. Each window stores a frozen strategy definition,
execution definition, version hash, cohort identifier, and
`prospective_no_backfill` evidence basis. Its starting valuation is zero by
construction. Reconstructed January or June slices remain useful recency views,
but they do not count toward this prospective cohort's proof.

Monitoring start dates are first-class evidence. The same strategy can have a
paper champion window, a later challenger window, and a fresh experiment-start
window. These start-date splits are deliberate. They make it possible to
distinguish "this strategy caught a favorable early-year window" from "this
strategy still behaves well from a fresh start."

Research Lab exposes that distinction inside the selected candidate view.
The Historical vs Running Experiment panel compares the current monitoring
window with the candidate's own historical 3-month forward behavior: median
strategy return, benchmark excess, drawdown envelope, false-alarm prior versus
similar-setup posterior, and whether the live or paper experiment has re-risked
below the defensive threshold. This is the bridge between backtest belief and
forward evidence. The user no longer has to infer the gap by bouncing between
Monitoring and Research Lab.

The Forward Test section turns recommendations into an audit trail. It supports:

- target/current book alignment,
- locked recommendation tickets,
- ticker-level target weights,
- size and price ranges,
- execution logging,
- paper or live mode,
- exact timestamps,
- execution notes,
- allocation history before and after the monitoring start.

Book Alignment answers a narrow but important question: is the selected paper or
live book already close enough to the current target, or does it need a small or
material rebalance? This prevents the top-line dashboard from repeatedly saying
"reduce risk" after the user has already logged paper trades that implemented
the prior recommendation.

Book Alignment is based on logged executions, not on recommendation tickets.
Locking a ticket records what the system suggested; it does not change the
tracked book. The book changes only after an execution is logged. The dashboard
can then recalculate alignment from the local journal and latest target. DuckDB
warehouse tables may lag until the journal is migrated, so monitoring and
audit-table views should be read as warehouse snapshots while Forward Test is
the operating surface for the freshest logged-book state.

The boundary between Launch Lab and Forward Test is important. Launch Lab is for
new or scale-up capital before it becomes an actively monitored sleeve. Once a
sleeve is running, Forward Test and Book Alignment become the operating source
of truth. At that point the question is no longer "should I launch this
strategy?" but "is my paper or live book aligned with the selected target, and
do I need a ticket?"

The system also includes a taxable-account framework. Taxable modeling is more
complex than IRA-style trading because turnover can create realized gains,
short-term tax drag, wash-sale concerns, and tax-lot consequences. Trade Bot's
taxable layer is an estimate, not tax advice, but it allows after-tax utility,
lot reconstruction, realized gain/loss estimates, and tax-aware warnings to be
considered when a strategy is evaluated outside a tax-advantaged account.

The operating standard is paper-first. A strategy can be researched, promoted to
paper monitoring, compared against challengers and references, and only then
considered for small real-money testing. Scaling should depend on forward
evidence, not only backtest confidence.

## 10. Dashboard Design And User Workflow

The dashboard is organized to answer questions in the order a human needs them.
The top of the page is not a research dump. It starts with the action headline,
operating brief, and book alignment. That answers "what do I need to do today?"
before showing the deeper evidence.

Dashboard V2 branches into focused sections:

- **Today**: action headline, operating brief, current posture, and book-aware
  recommendation.
- **Macro**: macro driver state, raw macro/ticker explorers, signal tables, and
  external narrative comparison where local transcript artifacts exist.
- **Risk**: portfolio risk engine, operating exposure, instability, scenario
  lattice, confirmation matrix, market health, and momentum diagnostics.
- **Forward Test**: tickets, execution logs, latest book alignment, and
  allocation history.
- **Research**: outcome frontier, Cycle Tracker, candidate deep dive, QC
  diagnostics, false-alarm judgement, and live-versus-history experiment
  comparison.
- **Performance**: historical performance and custom windows.
- **Launch**: entry-gate evidence for new or scale-up capital, including staged
  launch protocols, horizon sensitivity, and experiment-operator contracts.
- **Simulation**: strategy simulations, validation calibration, and full
  simulation workbench.
- **Monitoring**: champion/challenger/reference forward evidence by start-date
  cohort.

The Command Center also includes a Change-Over-Time station for the operating
metrics where direction matters. It shows compact trend charts for risk score,
1-month risk-off probability, risk-budget multiplier, risk constraints, regime
instability, macro drivers, monitoring evidence, and simulation quality. This
helps distinguish "the system just changed today" from "the system has been
operating defensively for weeks." The history is powered by local snapshot and
operating-history retention rules: recent periods can be kept at higher
granularity, while older backfilled history can be thinned to weekly snapshots
to avoid uncontrolled local data growth.

The News & Macro workflow includes an external macro comparison lane. Public
42 Macro videos can be stored as local video metadata and transcripts, classified
for tactical risk posture, compared with point-in-time Trade Bot operating
history, and scored against subsequent market outcomes where horizon data is
available. This is a disagreement and outcome audit, not an attempt to clone a
proprietary macro model. Transcript coverage, YouTube access limits, and the
fact that many macro claims are long-horizon or qualitative remain important
caveats. The useful output is narrower: when Trade Bot and an external human
macro process disagree during large-change moments, the system can preserve that
record and later ask which posture handled the next 1 week, 1 month, or 3 months
better.

The July 21, 2026 refresh covered 256 transcript-backed videos through the same
day. An audit found and fixed a material comparison bug: the old mapping treated
the 90% risk-budget capacity as if it were total risk exposure, causing the
then-current 63.73% defensive Trade Bot portfolio to be labeled risk-on. The canonical mapping now
uses final defensive allocation. On that basis, the newest 42 Macro transcript
is `constructive_but_fragile` (-0.15) and Trade Bot is `cautious` (-0.27), an
aligned 0.13-point gap. The qualitative agreement is strongest on extreme
dispersion, leverage/concentration, AI-capex fragility, and credit as the
confirmation channel. The horizon read is not identical: 42 Macro still calls
the present regime risk-on and expects crowded positioning to make a short
squeeze the likely first move, whereas that contemporaneous Trade Bot snapshot
held only 36.27% risk exposure. Because Trade Bot's news, event, and scenario allocation
authorities are zero, this sizing is not a reflection of the 42 Macro content,
although both processes observe some of the same market-price evidence.

The expanded July 14-21 review also exposed a limit in the comparison itself:
the transparent keyword classifier can collapse mixed horizons incorrectly. It
called the July 15 discussion `risk_on` even though the human read was a mixed
cross-asset structural warning, not a clean equity allocation call. The current
conclusion therefore uses the videos' horizon-specific substance: 42 Macro is
more constructive over the next days or weeks because crowded shorts may support
the index, while both systems are cautious over the next one to two quarters
about AI-capex deceleration, dispersion, concentration, Fed uncertainty, and
credit confirmation. Historical scalar scores remain screening proxies, not
reconstructed 42 Macro positions.

The right-side quick-reference panel explains terms, metrics, tickers, and
workflow objects. This matters because the system uses many concepts that are
easy to misuse. For example, max drawdown, Ulcer Index, recovery return needed,
beta-adjusted S&P delta, and time below prior peak all describe different
things. A user should not need to leave the dashboard to understand what a
metric means or how it can mislead.

The left sidebar controls daily operations. Most routine refreshes can be run
from the UI: full daily update, snapshot rebuild, warehouse migration, paper
valuation, monitoring-window seeding, and ML diagnostics. The dashboard also
exposes local refresh controls for book alignment after executions are logged.
Long experiment sweeps, dependency changes, Git operations, and any live-broker
activity remain outside the one-click path because they require explicit human
intent and review.

## 11. Governance, Limitations, And Appropriate Use

Trade Bot is an evidence system, not an oracle. It can reduce ambiguity, expose
tradeoffs, and enforce a paper-first process, but it cannot remove uncertainty.
The main limitations are:

- Historical backtests can overfit.
- Public data proxies are imperfect.
- Macro data can revise.
- News and narrative signals can be compelling but untestable.
- Regime labels are simplified representations of complex markets.
- Forward simulations depend on historical analogues that may not repeat.
- Human execution can differ from idealized target weights.
- Historical universe membership and delisting evidence are not yet verified
  for the current research corpus.
- The persisted trial ledger cannot recover experiments that never wrote a
  manifest or durable candidate artifact.

The system tries to address these limitations through explicit model authority,
validation gates, walk-forward tests, driver rotation, factor attribution,
simulation diagnostics, paper monitoring, and execution journaling. The point is
not to eliminate judgment. The point is to make judgment more disciplined,
traceable, and falsifiable.

The most appropriate use is iterative:

1. Refresh the daily snapshot.
2. Read the action headline and book alignment.
3. Review quantitative risk and hard portfolio constraints if action is
   non-trivial; read scenarios as research context unless their displayed
   authority is non-zero.
4. Check monitored strategy evidence.
5. Use Launch Lab before putting new or scale-up capital into a selected
   strategy.
6. Lock and log paper recommendations before treating them as followed.
7. Promote, demote, or archive strategies based on both historical and forward
   evidence.
8. Keep narrative inputs in the right authority lane unless ablation tests or
   market confirmation promote them.

In that sense, Trade Bot is less a single model and more a research operating
system for tactical allocation. Its value comes from the combination of
strategy testing, risk-aware sizing, calibration-gated scenario planning, paper
monitoring, and auditability.

## Conclusion

Trade Bot's central insight is that a usable trading system needs both
quantitative evidence and operating discipline. A strong strategy is not just a
good equity curve. It must have interpretable mechanics, tolerable drawdowns,
validated signals, reasonable turnover, forward monitoring, and a clear process
for translating targets into human-reviewed actions.

The research to date points toward growth-seeking strategies with disciplined
off-ramp and re-entry logic as the most promising family. Purely defensive
systems are often too low return for the objective, while simple buy-and-hold
can leave the user exposed to large concentration-driven drawdowns. The best
current candidates attempt to keep enough upside to matter while reducing the
left-tail events that can derail compounding or cause behavioral failure.

The system is deliberately local, paper-first, and review-oriented. It is built
to keep improving as new data, experiments, and forward monitoring results
arrive. The right standard is not whether any one backtest looks impressive.
The right standard is whether the process continues to identify robust,
human-executable strategies and then tracks them honestly in the real world.
