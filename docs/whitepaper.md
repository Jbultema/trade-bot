# Trade Bot System Whitepaper

Status: canonical overview. Last reviewed: 2026-07-06.

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
sanity checks to reduce severe drawdowns and re-enter after stress. Current
research artifacts show leading candidates in the approximate range of 14-16%
CAGR with max drawdowns near 20-23%, depending on the exact model and validation
gate. Those numbers are historical research outputs, not forecasts.

Trade Bot is also built around the reality that implementation matters. A
strategy that looks good in a backtest still has to be monitored from a specific
start date, valued forward, translated into target weights, logged as tickets,
and reconciled against the user's actual paper or live book. The system therefore
includes champion/challenger monitoring, paper valuations, book alignment,
recommendation tickets, execution logs, taxable-account estimates, and a
Forward Test area. The goal is to move from "interesting research" to a repeatable
process that can be reviewed, challenged, and audited before capital is scaled.

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
| Current-state engine | Builds the daily market read: risk status, confirmation matrix, regime pulse, scenarios, drivers, and current posture. |
| Strategy engine | Constructs target weights for baselines, research candidates, and selected operating systems. |
| Backtest engine | Applies execution lag, rebalance cadence, transaction costs, target weights, and equity compounding. |
| Risk engine | Applies scenario-aware sizing, expected shortfall, stress loss, factor exposure, concentration checks, and defensive floors. |
| Research Lab | Compares experiments, strategy families, outcome frontier, validation, factor attribution, and candidate details. |
| Simulation Lab | Projects selected strategies through deterministic, bootstrap, and regime-conditioned forward paths. |
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
- **Risk score**: a numeric summary behind the status. Higher is generally more
  constructive; lower is more defensive.
- **Confirmation matrix**: a collection of market, credit, breadth, trend,
  volatility, macro, and related checks that can be bullish, neutral, or bearish.
- **Scenario lattice**: probabilities for possible future market regimes across
  horizons such as 1 week, 1 month, 3 months, and 6 months.
- **Driver rotation**: a view that separates historically proven drivers,
  currently active drivers, emerging drivers, fading drivers, and explainer-only
  context.

The scenario layer does not claim to predict exact index levels. It maps current
conditions into broad future-state probabilities. Examples include risk-off,
transition, broad risk-on, fragile risk-on, or risk-off-then-relief. The sizing
engine uses these probabilities to adjust risk budgets and defensive floors.

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
volatility targeting, or scenario pressure. Its risk control may impose
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
- scenario-weighted downside risk,
- minimum defensive allocation.

A separate decision-sanity overlay exists because news and narrative pressure
can make a system too bearish. The overlay can cap event-only defensive moves
unless enough market-confirmed breaks appear. That is especially important in
fragile bull markets where the narrative risk is real but the market has not
yet confirmed a broad break.

## 5. What The Research Has Found About Strong Strategies

The most useful pattern so far is not "always own QQQ" and not "hide in cash."
The best historical candidates have generally combined high-growth participation
with drawdown-sensitive off-ramps and re-entry logic. The tradeoff that matters
for the intended use case is not maximum smoothness. It is high terminal wealth
with drawdowns that are painful but survivable.

Current research artifacts show several high-ranking candidates near the
growth-constrained frontier. Examples include re-entry volatility target
variants and high-CAGR AI/growth escape variants. Leading scorecard rows have
shown approximate CAGRs in the 14-16% range, max drawdowns around 20-23%, high
walk-forward positive rates, and average turnover around one trade event per
week or less. These are backtested and validated research results, not live
performance promises.

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

## 6. Testing And Validation Framework

Trade Bot uses several layers of testing because no single backtest statistic is
trustworthy enough on its own.

The first layer is the basic historical backtest. It applies target weights,
signal lag, rebalance cadence, transaction costs, and portfolio compounding.
This produces returns, CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar,
turnover, Ulcer Index, allocation history, and transaction behavior.

The second layer is rolling-window and regime testing. The system evaluates
performance across shorter windows such as 1 year, 3 years, and 5 years, as well
as calendar-year and market-regime slices. This matters because a strategy that
only works in one historical era is not enough. The user can inspect periods
such as 2008, 2011, 2018, 2020, 2022, and recent AI-led markets to understand
where a strategy exited, stayed defensive, re-entered, or failed.

The third layer is walk-forward validation. In walk-forward testing, the system
trains or selects based on one historical segment and evaluates on a later
segment. The purpose is to reduce overfit risk. A strong candidate should not
only have a good full-history score; it should have a positive walk-forward
rate, a tolerable worst rolling period, and reasonable performance in left-tail
regimes.

The fourth layer is outcome utility. For this project, the objective is not just
"high Calmar." The Growth-Constrained Outcome Frontier asks whether extra CAGR
is worth additional drawdown for a 15-year accumulation problem with ongoing
monthly contributions. It estimates terminal wealth, benchmark-relative wealth,
recovery return required after drawdown, Ulcer Index, drawdown penalties, and
validation confidence. This makes it possible to compare an 11% CAGR, -15%
drawdown system against a 15% CAGR, -22% drawdown system in terms of actual
wealth utility.

The fifth layer is ablation and signal evidence. The system can ask whether a
signal family improves CAGR, drawdown, re-entry, churn, or left-tail behavior
after costs. This is how narrative or newly added indicators should earn their
way into the action layer. If a signal cannot be tested or mapped to a relevant
now-casting framework, it should remain explanatory context.

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

The regime-conditioned model is the most distinctive. It is closer to a
scenario-aware Monte Carlo process than a simple CAGR calculator. Each path can
move through a sequence of regimes, such as risk-on, transition, fragile
risk-on, risk-off, or risk-off-then-relief. Strategy behavior is then sampled
from historical windows associated with those regimes. This produces ranges for
terminal wealth, drawdowns, severe-drawdown probability, and benchmark-relative
outcomes.

The Simulation Lab also includes interpretability. It should help answer:

- Does the simulated future resemble historical strategy behavior?
- Which regimes dominate the forward path distribution?
- Is the selected strategy outperforming references because of high median
  return, lower severe-drawdown risk, or favorable scenario weighting?
- Where is the model fragile because regime labels are coarse or data is sparse?
- How does the strategy compare to holding SPY or QQQ under the same contribution
  assumptions?

The simulation layer is planning support. It does not decide trades by itself.
It helps users understand whether a candidate's historical edge is plausible
under future-state assumptions and whether the range of outcomes is acceptable.

## 8. Monitoring, Tickets, And Making It Real

A common failure mode in research systems is stopping at the backtest. Trade Bot
adds the operational plumbing needed to make the research observable.

The Monitoring section creates champion/challenger/reference windows. A champion
is the main strategy currently being followed or most seriously evaluated. A
challenger is a competing candidate. A reference is a benchmark such as SPY, QQQ,
60/40, or cash-like exposure. Each window has a start date, paper capital, daily
valuations, cumulative return, benchmark comparison, drawdown, and forward
status. This is critical because paper monitoring from different start dates can
otherwise create misleading comparisons.

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
live book already close enough to the latest target, or does it need a small or
material rebalance? This prevents the top-line dashboard from repeatedly saying
"reduce risk" after the user has already logged paper trades that implemented
the prior recommendation.

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

## 9. Dashboard Design And User Workflow

The dashboard is organized to answer questions in the order a human needs them.
The top of the page is not a research dump. It starts with the action headline,
operating brief, and book alignment. That answers "what do I need to do today?"
before showing the deeper evidence.

The Insight Workbench then branches into focused sections:

- **Command Center**: current target posture and trade decision.
- **Risk & Scenarios**: risk engine, scenario map, stress, factors, and off-ramp
  logic.
- **News & Macro**: current context, driver rotation, and latest inputs.
- **Research Lab**: experiment comparison and strategy deep dives.
- **Simulation Lab**: forward path modeling and scenario-conditioned outcomes.
- **Performance**: historical performance and custom windows.
- **Monitoring**: champion/challenger forward evidence.
- **Forward Test**: tickets, execution logs, book alignment, and allocation
  history.

The right-side quick-reference panel explains terms, metrics, tickers, and
workflow objects. This matters because the system uses many concepts that are
easy to misuse. For example, max drawdown, Ulcer Index, recovery return needed,
beta-adjusted S&P delta, and time below prior peak all describe different
things. A user should not need to leave the dashboard to understand what a
metric means or how it can mislead.

The left sidebar controls daily operations. Most routine refreshes can be run
from the UI: full daily update, snapshot rebuild, warehouse migration, paper
valuation, monitoring-window seeding, and ML diagnostics. Long experiment
sweeps, dependency changes, Git operations, and any live-broker activity remain
outside the one-click path because they require explicit human intent and review.

## 10. Governance, Limitations, And Appropriate Use

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

The system tries to address these limitations through explicit model authority,
validation gates, walk-forward tests, driver rotation, factor attribution,
simulation diagnostics, paper monitoring, and execution journaling. The point is
not to eliminate judgment. The point is to make judgment more disciplined,
traceable, and falsifiable.

The most appropriate use is iterative:

1. Refresh the daily snapshot.
2. Read the action headline and book alignment.
3. Review risk/scenario context if action is non-trivial.
4. Check monitored strategy evidence.
5. Lock and log paper recommendations before treating them as followed.
6. Promote, demote, or archive strategies based on both historical and forward
   evidence.
7. Keep narrative inputs in the right authority lane unless ablation tests or
   market confirmation promote them.

In that sense, Trade Bot is less a single model and more a research operating
system for tactical allocation. Its value comes from the combination of
strategy testing, risk-aware sizing, scenario-conditioned planning, paper
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
