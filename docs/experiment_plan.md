# Experiment Plan

The system should expand breadth only after the baseline pipeline is reproducible. The first milestone is a strong current-state trade bot. Simulated-future strategy selection is a later milestone, not the next implementation target.

## Phase Boundary

### Phase 1: Current-State Trade Bot

The near-term system should answer: given current prices, trend, volatility, macro, credit, breadth, concentration, and relevant narrative evidence, what long-only action should be considered now?

Phase 1 priorities:

- reliable data ingestion and cached reproducibility
- current market regime and risk-state classification
- long-only strategy signals for brokerage-accessible stocks and ETFs
- risk management, position sizing, and drawdown controls
- walk-forward and regime-split backtesting
- human-readable dashboard and trade suggestion output
- live paper-trading log before any real-money scaling

Phase 1 should not depend on simulated future paths to produce recommendations.

### Phase 2: Simulated-Future-Enabled Trade Bot

The later system should answer: given the current state, what future states are plausible, how do strategies behave across those paths, and which current action has the best risk-adjusted expected utility?

Phase 2 priorities:

- probabilistic scenario generation
- macro/market state transition models
- return, volatility, correlation, and drawdown path simulation
- stress scenarios and left-tail state injection
- strategy policy testing across simulated futures
- decision rules based on expected return, drawdown risk, regret, and survivability

Phase 2 starts only after Phase 1 has a trusted backtest, dashboard, and paper-trading workflow.

## Validation Rules

- Use next-session execution assumptions by default.
- Keep a locked benchmark set: SPY, QQQ, VTI, and BIL.
- Report turnover and transaction costs for every strategy.
- Report full-history, calendar-year, and rolling-window performance. A single 2005-to-present score is never enough.
- Prefer walk-forward validation over single full-history optimization.
- Treat thresholds as policy-constrained hyperparameters, not values chosen solely because they won one backtest.
- Track strategy behavior by market regime and drawdown period.
- Evaluate "stay in versus pivot" behavior through shorter windows, including 1-year, 3-year, and 5-year rolling windows.

## Initial Expansion Queue

1. Absolute trend filters across SPY, QQQ, and VTI.
2. Relative ETF rotation across sectors and defensive assets.
3. Dual momentum with volatility targeting.
4. Drawdown-sensitive exposure scaling.
5. Breadth confirmation using cap-weight versus equal-weight proxies.
6. Credit confirmation using HYG/LQD, BIL/SHY/TLT, and FRED spreads.
7. AI-beta concentration dashboard.
8. Narrative transition index from curated news sources.
9. Earnings and event-risk exposure controls.
10. Walk-forward model tournament with holdout periods.

## Event-Risk And News Layer

Policy and geopolitical news should be converted into structured event families before it can
influence recommendations. The first deterministic layer supports:

- oil chokepoint and Iran/Hormuz supply shocks
- tariff and trade-policy shocks
- military escalation and de-escalation shocks
- AI unit-economics and capex-sustainability shocks
- AI infrastructure, power, and semiconductor-buildout shocks
- private-credit, direct-lending, and liquidity-transmission shocks
- energy-supply, oil-inventory, and commodity-inflation shocks
- sector narratives where the first signal is not a broad-index move
- policy-reversal risk, where the first market reaction may not persist

For each event category, the system should produce:

- a scenario playbook: what would confirm relief, escalation, or whipsaw
- affected tradable proxies: SPY, QQQ, RSP, IWM, XLK, IGV, SMH, SOXX, XLE, USO, BNO, GLD, TLT, HYG, LQD, BKLN, SRLN, BIZD, KRE, UUP, VIXY, BIL
- news-phase label: leading warning, coincident confirmation, lagging explanation, or uncertain
- off-ramp rules based on price confirmation, credit, breadth, oil, dollar, and volatility
- historical event-window diagnostics over pre-event, 1-day, 5-day, 21-day, and 63-day windows

The bot should not become a Trump-prediction engine. The proper abstraction is a
policy/geopolitical shock engine: classify the channel, compare it to historical analogs, monitor
market confirmation, and reduce risk when the event begins damaging credit, breadth, volatility,
or trend.

The same abstraction should handle sector narratives. A report on AI losses, private-credit
stress, grid/power bottlenecks, oil transport risk, or semiconductor export controls should be
classified by channel and phase before it touches position sizing. Some stories lead prices,
some arrive with price confirmation, and some explain moves already underway; the bot should
react more slowly to lagging explanations than to new evidence that is starting to confirm in
tradable proxies.

The broader experimentation program should keep a separation between alpha discovery, risk sizing, and human playbook outputs so the system remains debuggable.

## Current-State Dashboard Requirements

The dashboard should show both historical monitoring and current decision support.

Required sections:

- current risk status and risk score
- current trading-alert examples by strategy
- current holdings and material target-weight changes
- scenario sketch for risk-on continuation, choppy rotation, and risk-off break
- event-risk monitor with current news scenarios, confirmation checks, and historical analog windows
- granular future-state scenario lattice across 1-week, 1-month, 3-month, and 6-month horizons
- scenario drivers for trend, breadth, credit, AI concentration, volatility/liquidity, energy inflation, defensive pressure, duration, drawdown resilience, and style rotation
- vol-adjusted momentum signal table across the expanded universe
- confirmation matrix for breadth, credit, volatility, concentration, AI beta, and defensive assets
- data quality table so stale or short-history inputs are visible
- historical full-period, rolling-window, and calendar-year performance

The first implementation uses deterministic price/volatility/ratio signals. LLM-derived qualitative signals should be added later as auditable inputs to the same dashboard, not as a replacement for the numerical risk engine.

## Future-State Scenario Lattice

The first scenario engine is a deterministic prior generator, not a full simulation engine. It should
produce a granular scenario lattice that can later seed path simulations.

Each scenario row should include:

- horizon: 1 week, 1 month, 3 months, or 6 months
- scenario category: risk-on, AI concentration, credit, inflation/energy, rates/liquidity, policy event, defensive, or reflexive policy
- probability and rank within horizon
- expected bot posture
- preferred exposure and exposure to avoid
- confirmation, invalidation, and off-ramp rules
- severity and risk bucket

The lattice should be generated from observable drivers:

- market trend
- breadth
- credit
- AI leadership and concentration pressure
- volatility/liquidity
- energy/inflation relief
- defensive pressure
- duration support
- drawdown resilience
- style rotation

This is more useful than a three-state risk-on/chop/risk-off sketch because it can distinguish
between superficially similar states: a healthy broadening rally, a fragile AI melt-up, a
credit-led risk-off break, an oil shock, a rates squeeze, a defensive grind, or a fast drawdown
followed by a policy-put rebound.

## Universe Design

Use a broader signal universe than the tradable strategy universe.

- Signal universe: broad indexes, sectors, styles, credit, duration, commodities, currency proxies, volatility proxies, international ETFs, crypto proxies, AI-beta names, and mega-cap concentration names.
- Tradable strategy universe: a smaller brokerage-accessible subset selected by liquidity, account suitability, turnover, and backtest evidence.
- The expanded universe should feed risk, breadth, regime, and concentration monitors even if most assets are not active trade candidates.

## Gate Before Phase 2

The current-state bot should meet these conditions before the future-simulation engine becomes the primary focus:

- baseline strategies can be rerun from cache without manual cleanup
- walk-forward results exist for all candidate strategies
- rolling 1-year, 3-year, and 5-year diagnostics exist for all candidate strategies
- strategy results are broken out by major market regimes and drawdown periods
- risk overlays are evaluated separately from alpha signals
- dashboard shows current holdings, suggested trades, risk posture, and explanation
- paper-trading logs capture recommendation, rationale, execution assumption, and outcome

## Retirement Context

The bot is a side experiment in improving long-term retirement outcomes, not the core retirement plan. The operating objective is to test whether a forecasting and risk-management process can add incremental return or reduce left-tail risk against an already viable retirement trajectory.

This implies:

- avoid all-or-nothing allocation decisions
- require strong evidence before scaling capital
- prioritize capital preservation and survivability
- measure results against broad passive benchmarks and realistic account constraints
- treat the first live phase as paper trading or very small-dollar trading
