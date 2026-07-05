# Trade Bot FAQ

Status: canonical user-facing reference. Last reviewed: 2026-07-05.

This FAQ explains what Trade Bot is, what it is not, how to operate it, and how
to interpret common dashboard and research terms. It is written for users who can
operate a local Python app but may not know the project history.

## Quick Answers

### What is Trade Bot?

Trade Bot is a local decision-support system for long-only swing and momentum
research. It pulls market, macro, news, and event inputs into a daily snapshot,
builds a current-state risk read, suggests target allocations, tracks paper
strategies forward, and stores experiment evidence locally.

### Does Trade Bot place trades?

No. It does not connect to a brokerage account and it does not submit orders.
It generates recommendations, tickets, sizing guidance, and paper/live journal
records for a human to review.

### Is this day trading?

No. The system is built for human-triggered swing and momentum decisions. It
uses daily data, next-day signal lag in backtests, and trading cadence measured
in days or weeks, not minutes.

### What accounts does it support?

The first-class operating path is paper monitoring. It can also estimate IRA-like
pre-tax outcomes and taxable-brokerage outcomes. Taxable output is an estimate,
not tax advice or broker-grade lot accounting.

### Can it run on a normal laptop?

Yes. The app is designed to run locally with Python, Poetry, DuckDB, SQLite, and
Streamlit. Research batches can take time, but normal dashboard use should load
from the latest snapshot.

### What is the daily answer supposed to be?

The top of the dashboard should tell you:

- what changed,
- whether action is needed,
- what target posture is implied,
- whether the paper book is already aligned,
- what constraints or caveats matter,
- what to monitor next.

## Safety And Scope

### Is this investment advice?

No. It is research tooling and decision support. Users remain responsible for
their own investment decisions.

### Why is the system long-only?

The project is designed around retirement-account-compatible, human-reviewable
stocks and ETFs. Avoiding shorts, options, margin, and leverage makes the system
easier to execute, easier to audit, and less fragile operationally.

### Why does the app emphasize paper monitoring?

Backtests are necessary but not enough. Paper monitoring tests whether a strategy
still behaves sensibly after the research period, with current data, current
market structure, and the actual daily process a user would follow.

### What does "paper first" mean?

It means a strategy should be watched in a simulated account before any live
capital is considered. A paper window has a start date, capital base, role, and
daily valuation rows.

### Can multiple people use the dashboard?

Yes, but the current app is a local research tool, not a hosted multi-user SaaS
application. If sharing with users, keep it read-only unless they are trained
operators. Do not expose a local Streamlit process directly to the public
internet.

### What should never be committed to Git?

Do not commit `.env`, API keys, local caches, DuckDB files, SQLite journals,
large data exports, screenshots with private account details, or generated
reports unless intentionally creating a small fixture.

## Daily Operation

### What is the one command for a full daily refresh?

```bash
poetry run trade-bot run-daily-update
```

That refreshes prices, macro, news, the snapshot, the warehouse, and paper
valuations by default.

### How do I open the dashboard?

```bash
poetry run streamlit run src/trade_bot/dashboard/app.py --server.port 8501
```

Then open `http://localhost:8501`.

### What should I read first in the dashboard?

Read the top operating surface in this order:

1. Latest update timestamp under the header.
2. Action Headline.
3. Operating Brief.
4. Decision Brief if you need supporting context.
5. Book Alignment.
6. Insight Workbench only if you need deeper evidence.

### What does the Action Headline mean?

It compresses the current risk state, largest target change, news/event pressure,
scenario risk, open tickets, and next action into a single headline. It should
tell you whether the day is do-nothing, small-action, or critical-action.

### What is Book Alignment?

Book Alignment compares the latest target weights against the locally logged
paper or live execution book. It answers: "Did the user already act on the latest
recommendation, and is the book close enough to the latest target?"

### Why can the recommendation say "review reduce risk" but Book Alignment says "do nothing"?

The recommendation describes the current target posture. Book Alignment compares
that target against the locally logged book. If the book already reflects the
target within the configured drift band, no new ticket is needed.

### Why is the dashboard sometimes slow?

Use the sidebar default `Latest snapshot (fast)`. If `Live pipeline` is selected,
Streamlit recomputes the market pipeline inside the app session, which is much
slower.

### How do I know whether the dashboard is fresh?

The header includes a latest update strip showing snapshot timestamp, market
date, and risk state. The sidebar also shows the loaded snapshot.

## Recommendations And Tickets

### What is a recommendation ticket?

A ticket is a local record of a proposed action before execution. It captures the
strategy, ticker, action, target, price/size bands, rationale, and review status.

### Does locking a ticket execute a trade?

No. Locking records the recommendation as reviewed and ready for paper or live
execution. A human still places any real order outside the app.

### Where do I log paper trades?

Use the Forward Test workbench. It stores execution records in the local journal
and lets Book Alignment compare the logged book against the latest target.

### Why are price bands and size bands used?

They create an audit trail. If an execution happens far away from the intended
price or size, the shortfall/drift layer can flag it.

### What if I miss a trade window?

Do not backfill fake executions. Leave the ticket unexecuted or log what actually
happened. The point is to measure the real process, including missed actions.

## Strategy Research

### What is a strategy?

A strategy is a defined allocation rule. Some are simple references like SPY,
QQQ, and 60/40. Others are tactical systems that rotate between risk assets and
defensive assets based on momentum, drawdown, scenarios, risk constraints, or
other tested overlays.

### Where do strategies live?

Runtime baseline strategies are configured in `configs/baseline.yaml` and built
through `src/trade_bot/strategies/momentum.py`. Experiment-generated strategies
are built by `src/trade_bot/research/experiments.py` and stored as artifacts.
The dashboard reads both through snapshots and the DuckDB warehouse.

### What is the Research Lab?

Research Lab is the experiment and strategy evidence workbench. It has aggregate
views for comparing strategies and a candidate-detail workbench for inspecting a
single strategy.

### What is the difference between aggregate insights and candidate details?

Aggregate insights compare many strategies: leaderboards, curated shelf, outcome
frontier, family map, signal evidence, taxable impact, and validation/QC.
Candidate Details shows one selected strategy: explanation, performance,
allocation behavior, decision timeline, factor attribution, mechanics,
robustness, and manifest notes.

### What is a champion?

A champion is the main paper-monitored candidate for a given mode/account. It is
not automatically live-approved.

### What is a challenger?

A challenger is a paper-monitored strategy competing against the champion and
reference portfolios.

### What is a reference portfolio?

A reference portfolio is a baseline such as SPY, QQQ, BIL/cash, or 60/40. It
keeps tactical strategies honest.

### What does "promoted" mean?

Promoted means the research system thinks the candidate deserves further
monitoring or evolution. It does not mean the strategy should receive live money.

### What does "pruned" mean?

Pruned means the strategy is hidden from default operating views because it was
weak, redundant, research-only, not reconstructable, too noisy, or not useful for
daily decisions. It may still exist in archived research views.

## Metrics

### What is CAGR?

CAGR is compounded annual growth rate. It converts total return into an annualized
growth rate. Higher is better only after checking drawdown, turnover, regime
robustness, taxes, and benchmark context.

### What is max drawdown?

Max drawdown is the worst peak-to-trough decline during the test. A strategy with
high CAGR but unacceptable drawdown may be difficult to hold or scale.

### What is Calmar?

Calmar is CAGR divided by absolute max drawdown. It rewards return per unit of
worst drawdown, but it can over-favor conservative strategies that do not meet a
long-term growth objective.

### What is Ulcer Index?

Ulcer Index measures the depth and persistence of drawdowns. It is often more
informative than simple days-below-peak because it penalizes long, painful
drawdowns more than tiny daily dips below a prior high.

### What is days below prior peak?

It is the percentage of test days where equity is below its own previous high.
This can be high even for profitable strategies because most compounding paths
spend many days slightly below recent peaks.

### What is recovery needed?

Recovery needed is the return required to recover from the max drawdown. For
example, a -20% drawdown needs +25% to return to the prior high.

### What is the Outcome Frontier?

Outcome Frontier plots CAGR versus max drawdown and overlays terminal wealth,
utility tiers, and Pareto-efficient candidates. It asks whether extra CAGR is
worth the extra drawdown for a configured accumulation horizon.

### Is the 15-year wealth output a Monte Carlo forecast?

The headline 15-year wealth card is deterministic planning math: historical
CAGR applied to the configured starting account and annual contributions. The
selected-strategy section adds a historical block-bootstrap simulation that
resamples daily return sequences and reports P10, median, and P90 terminal
wealth plus simulated drawdown pain. That is better for sequence risk, but it
is still not a calibrated regime-conditioned forecast.

### Where do I change the 15-year outcome assumptions?

The defaults live in `src/trade_bot/DEFAULTS.py`: planning horizon, starting
account value, annual contribution, drawdown bands, bootstrap path count, and
bootstrap block length. After changing them, rerun the daily/experiment refresh
so stored scorecards and dashboard snapshots use the same assumptions.

### What is growth-constrained utility?

It is an outcome score that rewards projected 15-year wealth with contributions
while penalizing drawdowns beyond the preferred band, hard drawdown breaches,
weak walk-forward evidence, poor left-tail behavior, overfit risk, and churn.

### Why can a lower Calmar strategy still be preferred?

For a long accumulation account, a strategy with 14-15% CAGR and -20% to -22%
drawdown can produce more terminal wealth than an 11% CAGR strategy with -15%
drawdown, if the larger drawdown is behaviorally and financially tolerable.

## Risk And Scenarios

### What is risk status?

Risk status is a current-state label derived from market health and confirmation
signals. It is a high-level read, not a trade by itself.

### What is risk budget?

Risk budget is the allowed exposure after scenario pressure, event pressure,
macro pressure, and portfolio-risk constraints. A lower budget means smaller risk
asset weights or larger defensive weights.

### What are scenario probabilities?

The scenario lattice assigns probabilities to possible near-term market states
such as risk-off, transition, broad risk-on, narrow AI-led melt-up, and
risk-off-then-relief. It informs sizing and watch items.

### Does the system forecast prices?

Mostly no. It forecasts or estimates future-state probabilities and then uses
tested allocation rules, risk constraints, and backtests to decide whether those
probabilities improve outcomes.

### What is decision sanity?

Decision sanity prevents unsupported narrative or news pressure from forcing
large defensive moves unless confirmed by price, credit, volatility, breadth, or
trend. It is tested as an overlay, not assumed correct.

### What is Driver Rotation?

Driver Rotation separates normally important drivers, currently active drivers,
emerging drivers, and fading drivers. It helps users see whether the market is
being driven by proven allocation inputs or by context-only narratives.

### Are news stories trade drivers?

Usually no. News and narrative items are context unless ablation tests or market
confirmation promote them into a model driver. This prevents trading from vibes.

## Data And Sources

### What data is used?

The app uses Yahoo Finance-compatible market proxies, FRED macro series, local
news source configurations, curated event definitions, and locally generated
research artifacts.

### Does the system have Bloomberg-grade data?

No. Some commercial macro services use proprietary data that this local app does
not have. When a signal cannot be supported by available data or historical
proxies, it should stay in context/watchlist views.

### Why track so many tickers?

The ticker universe provides proxies for broad equities, sectors, factors,
duration, credit, commodities, currencies, volatility, crypto proxies, AI beta,
private credit proxies, and defensive assets. Not all tickers drive trades.

### Why is some data classified as thin proxy or unsupported?

Thin proxies are imperfect public substitutes for concepts like private credit,
AI capex stress, or equity supply. Unsupported items are useful reminders but do
not have enough data support to drive allocation.

## ML And Statistical Models

### Does Trade Bot use LLMs?

The core trading, backtesting, risk, and scoring logic is Python data science.
News/event text can be categorized by lightweight non-LLM logic. LLMs may be used
outside the app to help reason about narratives, but LLM text should not directly
drive trades without testable signals.

### Where is classical ML used?

ML diagnostics test future-state probabilities, feature importance, drift, and
classification performance. These are artifact-backed research layers, not
dashboard cold-start training jobs.

### Why are some ML strategies low CAGR?

Many ML risk classifiers are conservative. Their job may be left-tail reduction,
not return maximization. If they cut CAGR too much, they should remain diagnostic
or be rejected.

### What is the role of Bayesian thinking?

The system uses probabilistic scenario framing and can incorporate Bayesian-style
updating concepts, but the operating implementation is deliberately constrained:
probabilities must improve backtested or paper-monitored outcomes before they
become important.

## Taxable Accounts

### Can I use this for a taxable brokerage account?

Yes, as an estimated research layer. The app can estimate after-tax CAGR, tax
drag, realized short- and long-term gains, wash-sale warnings, and tax-loss
harvesting candidates.

### Is taxable output tax advice?

No. It is an estimate for research and planning. Real taxable-account use needs
broker lot reconciliation and professional tax review.

### Why might a good IRA strategy be poor in taxable?

High turnover and short-term gains can erode after-tax returns. Taxable Impact
helps identify whether the edge survives taxes.

### Should taxes stop a risk-off exit?

Usually no. Tax drag matters, but a real left-tail exit should not be ignored
just to defer taxes.

## Troubleshooting

### Port 8501 is already in use. What do I do?

Use another port:

```bash
poetry run streamlit run src/trade_bot/dashboard/app.py --server.port 8502
```

### The dashboard opens but data looks old.

Run a daily update:

```bash
poetry run trade-bot run-daily-update
```

Then refresh the browser.

### Monitoring does not update.

Run:

```bash
poetry run trade-bot migrate-warehouse
poetry run trade-bot run-paper-valuation
```

### A strategy is visible in research but missing from monitoring.

It may be research-only, archived, pruned, or not reconstructable from current
snapshot artifacts. Inspect it in Research Lab before trying to monitor it.

### A ticker lookup chooses the wrong term.

Exact ticker matches should rank above broader concepts. If `SMH`, `SOXX`, `QQQ`,
or another exact ticker does not resolve to the ticker card, check
`src/trade_bot/dashboard/ticket_explainers.py` and `src/trade_bot/DEFAULTS.py`.

### A command uses the wrong Python.

Check:

```bash
python --version
poetry run python --version
poetry env info
```

Then re-run setup from `docs/setup_guide.md`.

### Something broke after moving the repo.

Start a new terminal and Codex session rooted at the actual repo path. Check:

```bash
pwd
git status --short
poetry run pytest -q
```

### How do I know whether to trust a strategy?

Do not rely on one metric. Check:

- CAGR versus SPY and QQQ,
- max drawdown,
- Ulcer Index,
- worst rolling 1Y/3Y periods,
- walk-forward positive rate,
- left-tail regime return,
- factor attribution,
- turnover and action cadence,
- tax drag if relevant,
- paper monitoring results.

## Governance

### When should a strategy be promoted to paper monitoring?

When it has strong enough historical performance, tolerable drawdowns, walk-forward
support, explainable mechanics, reasonable turnover, and an implementation that
can be valued forward.

### When should a strategy be removed from default views?

When it is low-CAGR, overly defensive, redundant, too twitchy, failed validation,
research-only, not reconstructable, or not useful for the current operating
surface.

### When should a strategy receive live money?

Only after sustained paper monitoring, clear execution discipline, acceptable
drawdowns, no major drift from backtest behavior, and explicit human review.

### What is the minimum viable daily discipline?

1. Run the daily update.
2. Confirm the timestamp.
3. Read Action Headline and Operating Brief.
4. Check Book Alignment.
5. Review Monitoring.
6. Lock/log tickets only if action is warranted.
7. Do nothing when the system says no material action is needed.
