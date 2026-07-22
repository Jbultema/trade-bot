# New Chat Seed Prompt: I111 Adversarial Research

Status: copy-paste prompt. Last reviewed: 2026-07-20.

Use this prompt to seed a new chat with the latest 5.6 model on the highest
available settings.

```text
You are continuing work in the migrated trade-bot repo. Use:

/Users/bultemj/repos/trade-bot

Do not use the older repo at /Users/bultemj/repos/jarred/trade-bot.

Goal:

Do the best possible next evidence pass on the i111 native risk-repair /
AI-concentration strategy research. V2.2 execution hardening and the V2.3 fixed
smoothing slate are complete and produced no qualifying replacement. Focus on
cross-sectional exit/replacement behavior, prospective evidence for already
eligible operating candidates, and simulation-validation performance. Preserve
the prior top-tier performance area, roughly 22% CAGR with max drawdown around
-20%, without reopening completed threshold or smoothing sweeps.

- The system can get too defensive too early and stay defensive too long.
- AI concentration is the return engine, so blunt AI caps usually destroy the
  performance we are trying to preserve.
- Native strategy rules are preferred over post-strategy overlays.
- The current native challenger is promising, but still fragile under execution
  and AI-crash stress.

Start with repo hygiene:

1. Run `git status --short --branch`.
2. Do not revert unrelated dirty files.
3. Work directly on `main`; this is a personal project and the current V2.2/V2.3
   work should be built on or replaced, not stashed merely for isolation.
4. Review `docs/risk_repair_research.md` first.
5. Review the latest artifacts listed below before making new claims.

Current native challenger:

`i111_native_risk_repair_guard17_relief85_ai85_div`

It is implemented as the native `dual_momentum_risk_repair` strategy family in
`src/trade_bot/strategies/momentum.py` and configured in both:

- `configs/baseline.yaml`
- `configs/active_trading.yaml`

Important current findings:

- Baseline-config adversarial validation still shows the native challenger as
  the best balanced base-backtest row: about 22.11% CAGR, -19.68% max drawdown,
  1.12 Calmar, and 65.55% average AI/growth exposure.
- The correct label is `promising_but_fragile`, not promote-now.
- Start-date sensitivity is not the main problem.
- Execution sensitivity is the main problem:
  - daily rebalance roughly 19.64% CAGR / -28.86% max drawdown in adversarial
    validation;
  - Monday rebalance roughly 19.12% / -29.85%;
  - five-day signal lag roughly 19.36% / -24.25%.
- Direct QC showed daily rebalance roughly doubles average turnover from about
  7.7% to about 15.9%, but the drawdown penalty is too large to explain as
  costs alone. Timing/path exposure needs investigation.
- Active/daily config is materially worse:
  - native challenger around 17.27% CAGR, -30.11% max drawdown, 0.57 Calmar;
  - candidate-family PBO around 90%;
  - all rows are research-only until execution review.
- Candidate-family PBO inside the adversarial roster was 1.43%, but canonical
  30-strategy PBO was 35.71%, OOS loss probability 0.00%, label
  `moderate_overfit_risk`.
- Sequence bootstrap for the native challenger showed about 8.39% 5-year
  annualized return p05, about 22.51% median, and about 16.40% probability of
  breaching a -25% drawdown over a 5-year bootstrapped path.
- Synthetic AI-crash stress labels every i111 candidate
  `high_ai_crash_exposure`; the native challenger current-weight stress was
  about -11.47%, and p05 historical-weight synthetic stress was about -30.98%.
- Hedge overlays did not produce a clean win. Treat overlays as diagnostics
  unless they become native, human-executable rules.
- The 50-combination orthogonal search and 250-iteration frontier search both
  found zero qualifying improvements. Higher-return rows widened drawdowns;
  tail-improving rows gave up too much return.
- V2.2 execution hardening found zero promotion-like mechanisms. Five- and
  ten-session holds improved the worst path but reduced Wednesday CAGR to about
  19.34% and 18.53%, respectively.
- The fixed V2.3 raw/EWM-5/Mean-10 smoothing slate found no passing transform.
  EWM-5 and Mean-10 reduced weekday variance but failed Wednesday CAGR and
  drawdown noninferiority; both are `research_only`, not prospective challengers.
- Simulation validation is currently a tooling gap. The first attempt failed
  because the latest snapshot did not include the native challenger. A fresh
  baseline snapshot fixed the roster, but full and bounded simulation
  validation runs remained compute-bound with no partial artifacts and were
  stopped manually.

Key artifacts to inspect:

- `docs/risk_repair_research.md`
- `reports/i111_adversarial_validation/summary.md`
- `reports/i111_adversarial_validation_active/summary.md`
- `reports/backtest_qc/summary.md`
- `reports/backtest_qc_i111_native/summary.md`
- `reports/pbo_diagnostics_i111_latest/summary.md`
- `reports/native_i111_risk_repair/summary.md`
- `reports/i111_frontier_search/summary.md`
- `reports/i111_orthogonal_search/summary.md`
- `reports/i111_execution_hardening/summary.md`
- `reports/i111_execution_smoothing/summary.md`
- `reports/prebreak_hindsight/`
- `reports/defensive_signal_audit/`

Useful commands already run:

```bash
poetry run trade-bot run-i111-adversarial-validation \
  --output-dir reports/i111_adversarial_validation

poetry run trade-bot run-i111-adversarial-validation \
  -c configs/active_trading.yaml \
  --output-dir reports/i111_adversarial_validation_active

poetry run trade-bot audit-backtest-qc \
  --strategy i111_native_risk_repair_guard17_relief85_ai85_div \
  --output-dir reports/backtest_qc_i111_native

poetry run trade-bot audit-backtest-pbo \
  --output-dir reports/pbo_diagnostics_i111_latest \
  --top-n 30 \
  --partitions 8

poetry run trade-bot run-i111-orthogonal-search --max-new-combinations 50

poetry run trade-bot run-i111-frontier-search \
  --max-iterations 250 \
  --checkpoint-size 20

poetry run trade-bot run-i111-execution-hardening

poetry run trade-bot run-i111-execution-smoothing

poetry run trade-bot build-snapshot --no-write-report
```

Do not claim full simulation validation is complete unless you actually make it
complete. The attempted commands were:

```bash
poetry run trade-bot validate-simulation-engine \
  --strategy i111_native_risk_repair_guard17_relief85_ai85_div \
  --output-dir reports/simulation_validation_i111_native \
  --ablation

poetry run trade-bot validate-simulation-engine \
  --strategy i111_native_risk_repair_guard17_relief85_ai85_div \
  --output-dir reports/simulation_validation_i111_native_quick \
  --horizons 3m,6m,1y \
  --paths 150 \
  --skip-ablation
```

Both were stopped manually after long compute-bound runtime with no partial
artifacts.

Best next work packages:

1. Open a new, fixed study of cross-sectional exit/replacement behavior during
   clustered AI stress. Diagnose which names replace NVDA/AMZN-like losers and
   whether causal confirmation can improve the bad paths without erasing the
   Wednesday edge. Do not reopen the completed smoothing-window sweep.
2. Add point-in-time universe and survivorship audits for the high-growth
   roster. Treat this as a separate evidence problem from execution smoothing.
3. Fix simulation validation performance or add progress/partial artifact
   writing. Then rerun the native challenger and top alternatives through the
   forward-simulation calibration gauntlet.
4. Start prospective paper monitoring only for candidates that already satisfy
   their retrospective gate contract. The current smoothing transforms do not.
5. Re-run PBO/QC/adversarial validation after each substantial mechanism, not
   after every tiny threshold tweak.
6. Update `docs/risk_repair_research.md` with any new durable conclusion and
   include exact commands/artifact paths.

Research posture:

- Be adversarial. The burden of proof is on the strategy.
- Prefer robust mechanism families over one lucky parameter row.
- Treat recent AI dominance as an explicit forward-looking uncertainty.
- Separate alpha, sizing, and risk management. Do not bury everything in a
  single threshold soup.
- Keep outputs local/private, long-only, and compatible with human-reviewed
  retirement-account workflows.
- If you make code changes, run targeted tests and ruff before final response.
- Final response should state what was run, what changed, what was found, what
  remains unproven, and where the artifacts live.
```
