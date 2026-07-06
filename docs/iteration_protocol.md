# Iteration Protocol

The research loop is sequential and promotion-based.

One iteration means:

1. Test 3-10 candidate ideas.
2. Compare full-history, rolling-window, turnover, and drawdown behavior.
3. Promote, evolve, or reject each candidate.
4. Design the next iteration from the result of the previous one.

The target operating end state is not a large strategy zoo. The live system should converge toward
1-3 operational systems:

- one core system for primary allocation
- optionally one defensive/risk overlay
- optionally one satellite sleeve if evidence supports it

## Experiment Artifact Roots

The project has both historical and active experiment roots:

- `reports/experiments/`: original historical experiment archive.
- `data/experiments_reset_v2/`: active reset-era archive with more readable
  strategy names and later ML/operating-system experiments.

Dashboard loaders prefer `data/experiments_reset_v2/` when it exists
locally. In that mode, "All approaches" means all approaches in the active root
plus configured baselines; it does not automatically merge the older historical
root. If a future workflow needs full cross-root investigation, add an explicit
root selector or merged archive view with an `experiment_root` column.

Do not overwrite old experiment roots. Add new roots for clean resets, and keep
archived roots available for audit.

## Promotion Rules

Promotion is based on a scorecard, not one metric. The legacy `promotion_score`
remains useful for research triage, while `growth_constrained_utility_score` is
the preferred outcome lens for accumulation-account selection because it rewards
15-year terminal wealth with contributions while enforcing survivable drawdown
and validation gates.

Core promotion criteria:

- credible CAGR against SPY/QQQ benchmarks
- materially better drawdown profile than buy-and-hold QQQ
- acceptable 1-year, 3-year, and 5-year rolling-window behavior
- acceptable walk-forward holdout behavior across sequential one-year test windows
- survivable total returns in named left-tail and market-transition regimes
- reasonable turnover for human-triggered swing trades
- acceptable growth-constrained utility when the strategy is intended for
  accumulation-account monitoring
- understandable exposure and failure modes

Automatic reject signals:

- max drawdown worse than the left-tail threshold
- deeply negative 3-year rolling windows
- weak walk-forward positive-window behavior
- left-tail regime total return worse than the regime-fragility threshold
- improvement that only comes from a narrow historical regime
- strategy mechanics that cannot be operated manually

Account semantics are part of the scorecard contract. Current results are
pre-tax / IRA-like. Taxable-account rankings should not be inferred from current
scorecards unless the estimated tax-lot and after-tax simulator fields are explicitly selected.

## Broad-Then-Deep Structure

The first stage should go broad before going deep. The current harness uses:

- iterations 1-3 for broad exploration across core, risk-control, thematic, defensive, and
  scenario-proxy ideas
- iterations 4+ for adaptive deepening from prior winners and evolvers
- iteration 21+ for candidate operating systems that combine alpha selection, risk sizing, and
  scenario-aware exposure throttles
- category-diverse parent selection so one hot category cannot monopolize the next batch
- saved candidate manifests so later iterations evolve exact prior strategy configs

Iteration 1 is a controlled baseline expansion:

- faster versus slower dual momentum
- concentrated versus diversified winners
- cross-asset broadening
- sector plus defensive rotation
- AI-beta satellite with escape rules
- factor rotation
- low-turnover absolute trend
- volatility targeting
- tighter drawdown throttle

Iteration 2 adds risk-adjusted ranking, inverse-volatility sizing, trend confirmation,
single-asset caps, global rotation, credit/rates rotation, commodity shock proxies, and AI
infrastructure rotation.

Iteration 3 adds scenario-proxy probes: AI bubble escape, AI capex infrastructure rotation,
private-credit stress, policy whipsaw, oil shock, defensive equity, mega-cap platform caps,
crypto/liquidity proxy exposure, reflation rotation, and defensive barbell trend.

Later iterations evolve promoted candidates, but they retain category diversity across the research
queue. This prevents early convergence on one recent-history winner before competing mechanisms
have been stress-tested.

Iterations 42-49 add an active-trading track. These candidates are still long-only and human executable, but they are tested with `configs/active_trading.yaml`: daily rebalance checks, one-day signal lag, and higher transaction-cost assumptions. They are scored on the normal robustness metrics and must also be interpreted through turnover and practical action frequency before any paper-monitoring promotion.

Iterations 50-54 add a final deep/wide curation pass. This pass intentionally tests missed mechanisms without adding an unbounded strategy zoo: canary/off-ramp cores, AI escape and AI infrastructure switches, credit/private-credit warning sleeves, policy/oil/geopolitical shock barbells, low-churn active systems, defensive equity, speculative-liquidity micro-sleeves, and final curated operating-system composites. These candidates remain long-only, cached-universe friendly, and human executable. Their purpose is to feed the curated top-25 shelf, not to force automatic promotion.

The curated shelf is the operational research queue. It anchors on validation-aware score, then adds family champions, operating-system candidates, and active probes so the dashboard does not over-concentrate on one recent historical winner. A shelf entry means inspect or paper-monitor first; it does not mean live-trade approval.

Iterations 21-40 add the first operating-system layer:

- scenario-position sizing overlays that cut risk exposure when market-trend, breadth, credit,
  volatility/liquidity, oil/inflation, drawdown, or AI-concentration proxies deteriorate
- curated operating-system candidates for AI escape, cross-asset guardrails, credit-first
  defense, global macro rotation, quality/low-vol equity, sector breadth, oil/policy shock
  barbells, AI infrastructure, and low-turnover trend defense
- walk-forward holdout summaries saved to `walk_forward_summary.csv` and fold-level diagnostics
  saved to `walk_forward_folds.csv`
- named-regime diagnostics saved to `regime_metrics.csv` and `regime_summary.csv`
- promotion scoring that separates raw performance from robustness score

Short crash windows should be judged by total return and drawdown, not annualized CAGR. Regime
CAGR remains useful context, but the left-tail reject gate uses regime total return so a short
crash is not over-penalized by annualization math.

## Benchmark Context

Every saved scorecard includes benchmark-relative context:

- excess CAGR versus SPY and QQQ
- drawdown improvement versus SPY and QQQ
- Calmar excess versus SPY and QQQ

Promotion decisions remain research triage, not live-trading approval. A candidate that reduces
drawdown but trails QQQ can still be useful as an overlay or sleeve, but it should not be mistaken
for a complete core operating system.

## Leakage Discipline

Risk overlays must be executable with information available at decision time. Volatility targeting
and drawdown controls use lagged scale factors; they cannot use same-day returns to avoid the day
that triggers the risk signal.

## Creative Track

Starting no later than iteration 3, each iteration should include at least one deliberately creative
candidate from [creative_strategy_backlog.md](creative_strategy_backlog.md). Creative candidates are
still constrained by the operating rules: long-only, human executable, and auditable. Their purpose
is to discover useful mechanisms, not to inflate the live strategy count.
