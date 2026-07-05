# Trade Bot Research Learnings

Status: maintained research summary. Last reviewed: 2026-07-05.

This document summarizes the durable lessons from the project so far. It is not
a claim that any strategy will continue to work. It is a guide to what the
research has suggested, what remains uncertain, and where future experiments
should focus.

## Executive Takeaways

The main finding is that simple buy-and-hold references remain difficult to beat,
especially QQQ in recent history. However, several tactical systems have shown a
promising pattern: keep much of the high-growth upside while materially reducing
the worst drawdowns. The best candidates are not the most conservative systems;
they are growth-seeking systems with disciplined off-ramps, re-entry logic, and
guardrails against concentration and narrative overreaction.

The most important practical lesson is that risk-off is only half the problem.
Many systems can reduce drawdown by moving defensive. The harder and more
valuable problem is re-risking early enough after damage has been contained. The
best research direction has therefore moved toward re-entry, recovery capture,
and growth-constrained outcome optimization rather than maximum Calmar alone.

The second major lesson is that narratives can be useful but dangerous. AI capex,
private credit, Treasury supply, and concentration are important context, but the
system should not trade directly from story alone. Narrative items belong in the
news/context and driver-rotation layers unless they can be mapped to testable
proxies and shown to improve outcomes.

## What Worked Best

### 1. Growth-seeking strategies with drawdown controls

The most promising candidates generally preserve exposure to strong risk assets
while applying bounded risk controls. They do not simply sit in cash. They accept
some drawdown in exchange for better long-run compounding.

Useful traits:

- high enough CAGR to matter,
- max drawdown inside a tolerable band,
- lower Ulcer Index than raw high-beta exposure,
- reasonable turnover,
- re-entry logic after drawdowns,
- explicit benchmark comparison versus SPY and QQQ.

### 2. Re-entry and buy-the-dip overlays

Research increasingly suggested that the system needed to re-risk faster after
market repair signals. Avoiding a drawdown is valuable, but missing the recovery
can destroy the edge.

Useful re-entry ingredients:

- drawdown depth,
- realized volatility compression,
- breadth improvement,
- credit stabilization,
- trend repair,
- AI/semis leadership when relevant,
- scenario transition from risk-off toward broadening.

### 3. Decision-sanity caps

The decision-sanity overlay was added because news/event pressure could make the
system too bearish. The useful rule is: large defensive moves should require
market confirmation, not only narrative concern.

This is especially important in environments where:

- AI narrative risk is loud,
- concentration is high,
- headlines are plausible but hard to quantify,
- credit and volatility are not confirming a real break.

### 4. Outcome utility instead of pure Calmar

For a long accumulation horizon, the best answer may not be the highest Calmar
strategy. A lower-CAGR, lower-drawdown strategy can be easier to hold, but it may
fall short of wealth goals. The Outcome Frontier reframed selection around:

- projected terminal wealth,
- contribution-aware accumulation,
- tolerable drawdown,
- recovery burden,
- walk-forward evidence,
- churn and overfit risk.

### 5. Factor attribution

Factor attribution made it easier to detect look-alike strategies. Several
apparently different candidates were largely variants of growth/AI beta plus
defensive timing. That does not make them useless, but it reduces the value of
monitoring too many similar challengers.

## What Was Less Useful

### 1. Overly defensive low-CAGR systems

Some strategies improved drawdowns but cut CAGR too much. For the project goal,
3-5% CAGR is not useful unless it is a cash substitute or temporary defensive
state.

### 2. Failed or weak ML routers

Some ML approaches reduced risk but produced returns too low for the objective.
They may still be useful as diagnostics, but they should not become operating
systems unless they improve terminal wealth and re-entry behavior.

### 3. Unvalidated narrative trading

Narratives such as AI bubble risk, private credit concerns, IPO supply, or
policy uncertainty can be directionally insightful but hard to test. Without
validation or market confirmation, they should remain context.

### 4. Too many monitored strategies

Monitoring too many strategies creates clutter and weakens decision discipline.
The more useful structure is:

- one champion,
- a small number of challengers,
- a few reference portfolios,
- broad research archive kept out of default operating views.

## Most Important Signal Families

The strongest recurring signal families have been:

- trend,
- breadth,
- credit,
- volatility,
- drawdown/re-entry,
- AI or growth leadership,
- concentration,
- rates/duration,
- commodities/inflation in certain regimes.

The most useful signals vary by regime. Credit and volatility matter most for
left-tail risk. Breadth and trend matter for staying risk-on or re-entering.
AI/growth leadership has mattered heavily in the recent period but cannot be
assumed to dominate forever. Commodities and rates become more important in
inflation or reflation transitions.

## Current Context Lessons

The system has repeatedly had to separate two ideas:

1. The market can remain risk-on and profitable.
2. The market can also be concentrated, unstable, and vulnerable to sharp
   transitions.

That is why the current design avoids binary "all in" or "all cash" logic. It
tries to size exposure based on scenarios, confirmation, and risk budget.

AI capex and mega-cap concentration are important because they can affect:

- earnings revisions,
- free-cash-flow expectations,
- valuation convergence,
- semiconductor demand,
- power/infrastructure beneficiaries,
- broad index concentration.

But those themes should be validated through measurable proxies:

- QQQ/RSP,
- SMH/SPY,
- RSP/SPY,
- breadth,
- credit,
- volatility,
- earnings-revision proxies,
- AI infrastructure tickers,
- mega-cap platform relative strength.

## Strategy Family Lessons

### AI/growth escape systems

Strengths:

- captured strong growth trends,
- reduced some left-tail damage,
- produced high outcome-utility candidates.

Risks:

- can still be a disguised AI/growth bet,
- may fail if AI itself is the drawdown source,
- can underperform if broadening rotates away from mega-cap tech.

### Broad re-entry systems

Strengths:

- focus on repairing markets and buying dips,
- may reduce the "stuck risk-off" problem,
- can be more diversified than narrow AI strategies.

Risks:

- falling-knife risk,
- false repair signals,
- possible underperformance in long, grinding bear markets.

### Sector and asset-class rotation

Strengths:

- can capture energy, cyclicals, quality, international, or broadening phases,
- reduces one-note AI dependence.

Risks:

- more moving parts,
- greater data-mining risk,
- harder explanation and monitoring,
- sector leadership can be noisy and one-off.

### Low-churn balanced systems

Strengths:

- easier to operate,
- better fit for human review,
- potentially lower tax drag.

Risks:

- may react too slowly,
- may not reach return objectives.

## Dashboard Learnings

The dashboard is most useful when it follows this structure:

1. top operating answer,
2. book alignment,
3. workbench navigation,
4. aggregate research,
5. candidate deep dives,
6. monitoring and execution audit.

The app became less useful when it showed too many numbers without conclusions.
The best views now explain:

- what to do,
- why,
- what changed,
- what is still true,
- what would invalidate the read,
- whether the book already reflects the recommendation.

## Data Lessons

Public data can support a useful system, but it has gaps:

- no full Bloomberg-style earnings revision history,
- imperfect private credit proxies,
- incomplete AI capex/unit-economics history,
- noisy news inputs,
- limited flow and positioning data,
- release-lag and revision issues in macro data.

The correct response is not to ignore these areas. The correct response is to
classify them honestly:

- validated allocation driver,
- validated context,
- explainer-only,
- unsupported watchlist.

## Taxable Lessons

Tax drag can materially change strategy ranking. High-turnover strategies that
look good in an IRA can be less attractive in taxable accounts. The taxable layer
is most useful for screening, not final tax accounting.

Promising taxable strategy traits:

- lower turnover,
- longer holding periods,
- high enough pre-tax edge,
- fewer short-term realized gains,
- ability to harvest losses without destroying thesis,
- avoiding small noisy trades.

## What To Keep Testing

The most valuable future work is not more random strategy variants. It is
targeted evidence:

- re-entry timing after drawdowns,
- broadening/source-of-funds rotation away from concentrated AI,
- factor attribution and independence,
- regime-specific strategy routing,
- concentration and dispersion stress,
- taxable-account survivability,
- implementation shortfall from paper/live execution,
- drift from backtest behavior.

## What To Prune

Default operating views should hide:

- failed experiments,
- low-CAGR defensive systems,
- unsupported watchlists,
- thin-proxy diagnostics unless selected,
- stale narrative modules,
- redundant variants,
- strategies that cannot be valued forward,
- old plans that no longer describe current system behavior.

Pruning does not mean deleting the research history. It means protecting the
daily operating surface from noise.

## Current Best Operating Posture

The project should be operated as:

- research-heavy,
- paper-first,
- long-only,
- human-reviewed,
- evidence-gated,
- skeptical of narrative,
- open to high growth when drawdown is tolerable,
- strict about forward monitoring before live use.

The end state should not be dozens of active strategies. It should be a small
number of understandable systems supported by backtests, forward paper evidence,
and clear operating rules.
