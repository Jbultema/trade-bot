# Risk-Repair Research

Status: maintained research note. Last reviewed: 2026-07-21.

This note consolidates the i111 risk-repair work so future experiments do not
confuse diagnostic overlays with promotable native strategy variants.

## Current Architecture

The canonical strategy-native implementation is `dual_momentum_risk_repair` in
`src/trade_bot/strategies/momentum.py`.

It starts from dual momentum, then applies two native weight-construction
controls before the normal backtest engine applies volatility targeting and
drawdown control:

- Defensive relief: releases a small part of BIL exposure when broad repair
  evidence is constructive and late AI stress is not extreme.
- Conditional AI concentration repair: caps aggregate AI/growth exposure only
  when AI stress evidence clusters. Excess can go to BIL or to a diversifier
  sleeve.

The neutral i111 candidate roster lives in
`src/trade_bot/research/i111_candidates.py`. Both the overlay lab and native lab
should import candidates from there.

## Current Challenger

The current persisted native challenger is:

`i111_native_risk_repair_guard17_relief85_ai85_div`

It is present in both `configs/baseline.yaml` and `configs/active_trading.yaml`.
It is not the primary strategy.

Configuration summary:

- Source family: r19 i111 late-guard dual momentum.
- Core universe: QQQ, SMH, SOXX, IGV, NVDA, AVGO, MSFT, META, AMZN, PLTR.
- Drawdown guard: 84-day equity lookback, -17% trigger, 65% risk multiplier.
- Defensive relief: release 15% of BIL above 85% when constructive evidence is
  present.
- AI protection: cap AI/growth at 85% only when AI stress score reaches 90%;
  route excess to SPY, RSP, GLD, and TLT.

Latest direct configured backtest readout:

- CAGR: 22.106185%.
- Max drawdown: -19.681954%.
- Calmar: 1.1232.
- Average AI/growth exposure: 65.550950%.

## Labs And Commands

Native lab, preferred for promotable strategy variants:

```bash
poetry run trade-bot run-native-i111-risk-repair-lab
```

Primary output:

- `reports/native_i111_risk_repair/summary.md`
- `reports/native_i111_risk_repair/strategy_metrics.csv`
- `reports/native_i111_risk_repair/variant_summary.csv`
- `reports/native_i111_risk_repair/rolling_windows.csv`
- `reports/native_i111_risk_repair/walk_forward.csv`
- `reports/native_i111_risk_repair/calendar_years.csv`
- `reports/native_i111_risk_repair/manifest.json`

Overlay lab, useful as a diagnostic but not as a promotion target:

```bash
poetry run trade-bot run-i111-risk-repair-lab
```

Primary output:

- `reports/i111_risk_repair/summary.md`
- `reports/i111_risk_repair/strategy_metrics.csv`
- `reports/i111_risk_repair/variant_summary.csv`

Orthogonal search lab, preferred for surveying new mechanism families before
more threshold tuning:

```bash
poetry run trade-bot run-i111-orthogonal-search --max-new-combinations 50
```

Primary output:

- `reports/i111_orthogonal_search/summary.md`
- `reports/i111_orthogonal_search/strategy_metrics.csv`
- `reports/i111_orthogonal_search/candidate_roster.csv`
- `reports/i111_orthogonal_search/variant_summary.csv`
- `reports/i111_orthogonal_search/rolling_windows.csv`
- `reports/i111_orthogonal_search/walk_forward.csv`
- `reports/i111_orthogonal_search/calendar_years.csv`

Frontier search lab, used for larger multi-mechanism sweeps across the next
research topics:

```bash
poetry run trade-bot run-i111-frontier-search --max-iterations 250 --checkpoint-size 20
```

Primary output:

- `reports/i111_frontier_search/summary.md`
- `reports/i111_frontier_search/strategy_metrics.csv`
- `reports/i111_frontier_search/candidate_roster.csv`
- `reports/i111_frontier_search/checkpoint_summary.csv`
- `reports/i111_frontier_search/family_summary.csv`
- `reports/i111_frontier_search/rolling_windows.csv`
- `reports/i111_frontier_search/walk_forward.csv`
- `reports/i111_frontier_search/calendar_years.csv`

Adversarial validation lab, used to challenge the current/recent i111 work
instead of finding another best parameter row:

```bash
poetry run trade-bot run-i111-adversarial-validation
```

Primary output:

- `reports/i111_adversarial_validation/summary.md`
- `reports/i111_adversarial_validation/strategy_metrics.csv`
- `reports/i111_adversarial_validation/robustness_summary.csv`
- `reports/i111_adversarial_validation/start_date_sensitivity.csv`
- `reports/i111_adversarial_validation/execution_sensitivity.csv`
- `reports/i111_adversarial_validation/ai_monitor_audit.csv`
- `reports/i111_adversarial_validation/overlay_metrics.csv`
- `reports/i111_adversarial_validation/research_artifact_audit.csv`
- `reports/i111_adversarial_validation/candidate_pbo_summary.csv`
- `reports/i111_adversarial_validation/sequence_bootstrap.csv`
- `reports/i111_adversarial_validation/synthetic_ai_crash.csv`
- `reports/i111_adversarial_validation/gap_audit.csv`
- `reports/i111_adversarial_validation/manifest.json`

Fixed-slate execution-smoothing lab, used to test the V2.3 raw,
EWM-5, and trailing-mean-10 slate without opening another parameter sweep:

```bash
poetry run trade-bot run-i111-execution-smoothing
```

Primary output:

- `reports/i111_execution_smoothing/summary.md`
- `reports/i111_execution_smoothing/candidate_metrics.csv`
- `reports/i111_execution_smoothing/schedule_summary.csv`
- `reports/i111_execution_smoothing/rolling_windows.csv`
- `reports/i111_execution_smoothing/promotion_gates.csv`
- `reports/i111_execution_smoothing/pbo_summary.csv`
- `reports/i111_execution_smoothing/manifest.json`

Additional latest adversarial outputs:

- `reports/i111_adversarial_validation_active/summary.md`
- `reports/backtest_qc_i111_native/summary.md`
- `reports/pbo_diagnostics_i111_latest/summary.md`

## Latest Best-Possible Pass: 2026-07-20

This pass was not another tuning run. It was an adversarial review of the
current native i111 challenger and the recent risk-repair research stack.

Completed checks:

- Rebuilt the latest baseline snapshot so snapshot-backed tooling can see the
  current native challenger.
- Ran baseline-config adversarial validation across start-date sensitivity,
  execution sensitivity, AI monitor quality, overlays, candidate-family PBO,
  5-year block bootstrap, synthetic AI-crash stress, and artifact completeness.
- Ran the same adversarial validation under `configs/active_trading.yaml`.
- Ran the direct Backtest QC gauntlet for
  `i111_native_risk_repair_guard17_relief85_ai85_div`.
- Ran canonical 30-strategy backtest PBO diagnostics.
- Attempted full and bounded forward-simulation validation for the native
  challenger. Both runs were stopped after extended compute-bound runtime with
  no partial artifacts, so simulation validation is now a tooling/performance
  gap rather than completed evidence.

Baseline-config readout:

- The native challenger remained the best balanced base-backtest candidate:
  22.11% CAGR, -19.68% max drawdown, 1.12 Calmar, and 65.55% average AI/growth
  exposure.
- Its review label is `promising_but_fragile`, not "promote now."
- The original start-date artifact reset indicator and portfolio state at each
  requested date, creating an artificial warm-up cash period. V2.2 now computes
  each strategy over full history and slices/rebases the carried-state result;
  the current `start_date_sensitivity.csv` is regenerated with
  `state_mode=carried_state`. Older reset-state conclusions should be ignored.
- Execution sensitivity was the main problem. Across the adversarial roster,
  74 execution-stress rows breached the failure threshold.
- For the native challenger specifically, daily rebalance lowered CAGR to about
  19.64% and widened max drawdown to about -28.86%; Monday rebalance was about
  19.12% / -29.85%; five-day signal lag was about 19.36% / -24.25%.
- The execution-hardening decomposition showed daily rebalance roughly doubled
  average turnover from about 7.70% to about 15.92%. Removing 5 bps transaction
  costs recovers most of the daily CAGR gap, while leaving most of the drawdown
  gap intact: costs explain return drag; timing/path exposure explains the tail
  deterioration.

Overfit and sequence-risk readout:

- Candidate-family PBO inside the adversarial roster was 1.43% with 0.00% OOS
  loss probability, which supports the narrow native family.
- The broader canonical 30-strategy PBO was 35.71%, with 0.00% OOS loss
  probability and a `moderate_overfit_risk` label. This is the more conservative
  read because it tests a wider candidate surface.
- The baseline sequence bootstrap gave the native challenger a 5-year annualized
  return p05 near 8.39%, median near 22.51%, and about 16.40% probability of
  breaching a -25% drawdown over a 5-year bootstrapped path.

AI-crash readout:

- The synthetic AI-crash stress labels every i111 candidate as
  `high_ai_crash_exposure`.
- The native challenger current-weight stress return was about -11.47%, and its
  p05 historical-weight synthetic stress was about -30.98%.
- This does not invalidate the strategy. It says the strategy's return engine is
  still AI leadership, so future AI-led drawdown risk has to be monitored and
  stress-tested explicitly rather than capped away blindly.

Active-trading config readout:

- The active/daily config materially degraded the i111 family. The native
  challenger fell to about 17.27% CAGR, -30.11% max drawdown, and 0.57 Calmar.
- Candidate-family PBO under active config was 90.00%, sequence risk worsened,
  and every candidate was `research_only_until_execution_review`.
- The active config should not be used to argue that daily or more responsive
  trading improves this family. It currently shows the opposite.

Forward-simulation gap:

- `validate-simulation-engine` first failed because the latest saved snapshot
  did not include the native challenger. A fresh baseline snapshot fixed the
  roster gap.
- A full default simulation validation with ablation, then a bounded 3m/6m/1y
  150-path run, both remained compute-bound with no partial output and were
  stopped manually.
- Next work should optimize or instrument rolling-origin simulation validation
  before treating it as part of the standard adversarial loop.

## Interpretation

The strongest finding so far is that blunt AI caps are not the right path for
the 22% CAGR family. AI concentration is the return engine, so protection needs
to be conditional, late, and small.

The useful native improvement was light defensive repair, not aggressive risk
flooring. The best tested candidate preserved the AI-heavy engine while nudging
the max drawdown below the previous top-tier target band.

The 50-combination orthogonal search tested signal speed, ranking quality,
concentration shape, broader US/factor/global universes, diversifier universes,
and AI-leadership confirmation gates. It found no considerable improvement over
the current native challenger. Higher-CAGR candidates existed, but their gains
came with worse drawdowns. The strongest non-promoted lead was the high-vol,
top-3 concentration path, which reached about 23.37% CAGR but widened max
drawdown to about -21.34%.

The 250-iteration frontier search tested five larger mechanisms and
combinations: confirmation-gated high concentration, dynamic guard selection, AI
leadership health scoring, crash-onset mesh, and a two-model router. It found
zero big improvements. Some rows passed a narrow preliminary screen, but none
displaced the current native challenger.

Frontier search readout:

- Current native challenger baseline: 22.11% CAGR, -19.68% max drawdown, 1.12
  Calmar, and 65.55% average AI/growth exposure.
- Best score rows were dynamic-guard variants that essentially matched the
  native challenger, but trailed by about 0.07 percentage points of CAGR.
- Best CAGR row was a gated-concentration variant at 23.43% CAGR, but max
  drawdown widened to -21.65%, so it failed the risk objective.
- Crash-onset mesh slightly improved max drawdown in some rows, down to about
  -19.30%, but gave up too much CAGR, with the best balanced row near 21.84%.
- Two-model router did not help: its best score row was about 22.33% CAGR with
  -20.62% max drawdown, and the higher-CAGR rows widened drawdown further.

The practical conclusion is that the current native challenger is still the best
balanced strategy in this family. The new mechanisms are more useful as
diagnostics and guardrail research inputs than as immediate promotable strategy
replacements.

The adversarial validation pass confirmed that conclusion, but tightened the
language. The native challenger is still the best base-backtest candidate at
22.11% CAGR, -19.68% max drawdown, and 1.12 Calmar. However, the correct
adversarial status is `promising_but_fragile`, not fully hardened, because
execution-stress assumptions and AI-crash exposure matter a lot. Two-day and
five-day signal lag, rebalance-day changes, daily rebalance, and high
transaction-cost stress can widen drawdowns materially.

Adversarial readout:

- Start-date sensitivity must be read from the carried-state artifact. The old
  reset-state minimum-CAGR statement is retired because it benefited from an
  artificial warm-up cash period.
- Execution sensitivity is the main problem to investigate next: the native
  challenger fell to about 19.36% CAGR / -24.25% max drawdown under five-day
  signal lag and about 19.64% CAGR / -28.86% max drawdown under daily
  rebalance.
- AI monitor evidence is useful but noisy. Four monitor/horizon rows cleared
  the useful-warning read, mostly around 63-day forward drawdown risk, but
  false-positive rates remain high.
- Hedge overlays did not produce a clean win. Conditional BIL overlays slightly
  improved max drawdown but gave up too much CAGR; GLD/TLT sleeves were mostly
  drag.

The execution-hardening and fixed-smoothing studies below close the first pass
on that gap. They confirm that timing/path exposure, not the drawdown guard, is
the central problem; neither study produced a promotable repair.

## V2.2 Execution Hardening: 2026-07-20

V2.2 added a dedicated execution-hardening lab:

```bash
poetry run trade-bot run-i111-execution-hardening \
  --output-dir reports/i111_execution_hardening
```

Primary outputs:

- `reports/i111_execution_hardening/summary.md`
- `reports/i111_execution_hardening/execution_variant_metrics.csv`
- `reports/i111_execution_hardening/mechanism_summary.csv`
- `reports/i111_execution_hardening/component_decomposition.csv`
- `reports/i111_execution_hardening/action_path_diagnostics.csv`
- `reports/i111_execution_hardening/calendar_year_comparison.csv`
- `reports/i111_execution_hardening/manifest.json`

The lab expanded the execution comparison to Monday through Friday, daily,
two-session lag, and five-session lag. It also decomposed the normal result
into full, no-transaction-cost, no-drawdown-guard, no-volatility-target, and
raw-weight paths.

Durable execution finding:

- The 22.11% CAGR / -19.68% max-drawdown result is specifically the historical
  Wednesday-lag-one path. It is not yet a robust live expectation.
- Tuesday and Monday were both near -29.9% max drawdown; daily was -28.86%;
  Thursday was -27.06%; and Friday was -24.78%.
- The difference is concentrated in the 2022 AI/growth unwind. Wednesday lost
  16.54% in 2022, while daily lost 26.65%, Monday lost 28.10%, and Tuesday lost
  27.57%.
- Removing the drawdown guard barely changed daily max drawdown (-28.86% with
  the guard versus -28.88% without it). The guard is not the root cause.
- Removing 5 bps transaction costs raises daily CAGR from about 19.64% to about
  22.06%, while max drawdown remains about -27.58%.
  Daily turnover is therefore a return-cost problem; weekday name selection is
  a separate tail-risk problem.
- Volatility targeting materially limits the absolute loss, but the remaining
  weekday sensitivity comes from cross-sectional name selection and replacement
  timing. NVDA and AMZN were the largest 2022 loss contributors across the
  fragile paths.

V2.2 native mechanisms:

- Added an optional `risk_repair_ai_cap_basis: risk_sleeve` mode. Unlike the
  original whole-portfolio cap, it measures AI concentration against the
  pre-repair active sleeve, so a large BIL allocation does not automatically
  make the cap dormant.
- Added native weight-path controls for minimum gross change, maximum step,
  minimum hold days, and a fast risk-off override. Defaults are inert and the
  configured challenger was not changed.

No mechanism qualified for promotion:

- A very late 50% active-sleeve AI cap preserved the Wednesday path at 22.23%
  CAGR / -19.30% max drawdown, but its worst execution drawdown was still
  -29.87%.
- A clustered 70% active-sleeve cap and an earlier 60% cap gave up return
  without consistently repairing the worst path.
- An 8% weight-change buffer was essentially neutral and did not harden the
  tail.
- A five-session hold improved the worst execution drawdown to -26.37%, but
  reduced Wednesday CAGR to 19.34%.
- A ten-session hold improved the worst execution drawdown to -22.93% and cut
  failure count from seven to one, but reduced Wednesday CAGR to 18.53%.

The next mechanisms should target cross-sectional exit/replacement behavior.
Earlier AI caps trade away the return engine without solving schedule
dependence, and the completed fixed causal-smoothing slate below did not clear
its noninferiority gates. PBO and full adversarial reruns were not used to
promote any V2.2 row because the bounded execution gauntlet produced zero
promotion-like candidates.

## V2.3 Fixed-Slate Execution Smoothing: 2026-07-20

The fixed V2.3 slate tests only three target paths: raw native i111, causal
EWM-5, and causal trailing-mean-10. Adding another smoothing window requires a
new study version rather than silently widening this candidate set.

Current results at configured 5 bps costs:

- Raw: 22.11% Wednesday CAGR / -19.68% drawdown; 19.64% median weekday CAGR;
  -29.87% worst weekday drawdown; 2.99-point weekday CAGR range.
- EWM-5: 20.62% / -23.73% on Wednesday; 20.41% median weekday CAGR; -25.44%
  worst weekday drawdown; 0.77-point range.
- Mean-10: 20.19% / -24.57% on Wednesday; 20.17% median weekday CAGR; -24.57%
  worst weekday drawdown; 0.77-point range.
- At 25 bps costs, median weekday CAGR was 15.09% raw, 16.70% EWM-5, and
  17.10% Mean-10.
- Family-level CSCV across 15 strategies and 70 train/test splits produced
  21.43% PBO and 0.00% OOS-loss probability. This is one family-level gate
  repeated on each transform row, not separate per-transform PBO evidence.

The completed screen has eight gates. Raw passed seven of eight but failed the
schedule-stability gate. EWM-5 and Mean-10 each passed six of eight: both
improved schedule stability and the worst execution tail, but failed Wednesday
CAGR and Wednesday drawdown noninferiority. Both therefore remain
`research_only`; neither qualifies to enter prospective paper monitoring. A
future transform must pass the fixed retrospective screen before a 63-126
session prospective trial is justified.

All V2.2/V2.3 labs now write `manifest.json` with config, price-frame identity,
Git/source-tree identity, environment, parameters, and per-artifact size/SHA-256
integrity records.
Reports remain explicitly retrospective and cannot promote automatically.

Research caution: no tested V2.2 or V2.3 mechanism is a primary-strategy
replacement. The empirical next path is a new, fixed study of cross-sectional
exit/replacement behavior, plus prospective monitoring of already-approved
operating candidates, rather than another smoothing-window or threshold sweep.

## V2.2 Reliability And Fixed Replacement Follow-Up: 2026-07-21

The next pass repaired operating semantics before running more strategy work:

- BIL is excluded from the risk-asset single-position cap, matching the actual
  sizing rule.
- Constraint comparisons use absolute and relative tolerance, and the headline
  summary now includes every hard constraint shown in the detail.
- Cache-fallback news is triage-only and retains its original data-as-of time.
- Events now support explicit decay and expiry; degraded cached news cannot
  create a sizing-authority event.
- A five-member prospective cohort was frozen at the 2026-07-21 market date:
  primary, native challenger, Min-25, QQQ, and SPY. Frozen definitions and
  version hashes are stored, and no earlier return counts as forward evidence.

The fixed cross-sectional replacement command is:

```bash
poetry run trade-bot run-i111-cross-sectional-replacement \
  --config configs/active_trading.yaml
```

The study fixed its design before execution. Clustered stress means a score of
0.75, or six of eight native AI-stress components. AI exits remain immediate;
only new or increased AI targets are redirected to BIL or RSP. No cutoff, cap,
hedge, or smoothing sweep was run.

Empirical read:

- Native reference: 21.01% Wednesday CAGR / -20.34% Wednesday drawdown;
  -30.60% worst execution-profile drawdown; seven failures.
- BIL deferral: 18.85% / -26.69%; -30.32% worst drawdown; eight failures.
- RSP deferral: 19.29% / -27.60%; -34.29% worst drawdown; eight failures.
- Both policies are `no_robust_improvement`. Neither enters the prospective
  cohort or changes the operating strategy.

The point-in-time/trial-governance pass wrote
`reports/research_governance/`. After including this study, the ledger contains
537 manifested completed rows across 11 study manifests, with no missing
declared roster among those manifests. All 537 rows still lack verified
point-in-time membership/delisting evidence, and unmanifested historical
attempts remain unknowable. Promotion is therefore fail-closed on this gate.

Rolling-origin simulation now writes an atomic CSV checkpoint plus an input and
configuration fingerprint. Resume skips completed origins deterministically;
changed inputs require `--restart`. A deliberately small native-i111 smoke test
completed 77 quarterly one-month origins at 20 paths each and then resumed
without recomputation. It showed 45.45% interval coverage, 4.57% median absolute
error, 3.90% launch-decision accuracy, and a 44.16% action score. The action
layer remains `action_checks_not_ready`. This proves checkpoint mechanics and
provides a warning signal; it is not the full multi-horizon, high-path
calibration previously blocked by runtime.

Current next step is evidence collection, not another mechanism sweep: allow
the frozen cohort to accumulate untouched observations, source a defensible
point-in-time universe/delisting dataset, and finish the now-resumable full
native-i111 calibration.
