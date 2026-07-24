# I111 Selector And Transition Repair

Status: pre-registered retrospective research. Allocation authority is zero.

## Question

Can a new i111 strategy repair repeatable cross-sectional selection and
transition failures without turning the strategy into a generic defensive
overlay or tuning specifically to the 2022 and 2023-24 misses?

The operating primary and native challenger are unchanged while this study is
run.

## Fixed candidate slate

The reference is the configured
`i111_native_risk_repair_guard17_relief85_ai85_div`. Five different native
mechanisms are compared with it:

1. `incumbent_buffer`: retain at most two held names when they remain inside the
   top six, retain positive absolute momentum, and a proposed replacement lacks
   at least a 15-percentile rank advantage.
2. `blended_rank_63_126`: select on a 65% short/35% slow percentile blend of
   risk-adjusted 63-session and 126-session momentum. Absolute eligibility still
   uses the configured 63-session minimum return.
3. `diversified_core15`: keep the native selector but reserve 15% of the active
   risk budget for an equal SPY/RSP core.
4. `recovery_meter_core`: allow exits immediately, but execute only half of an
   unconfirmed new/increased position at each decision. Confirmation requires
   positive 21-session recovery above 2%, price above its 50-session average,
   and SPY above its 100-session average. Deferred risk goes to SPY/RSP, not
   BIL.
5. `integrated_selector_transition`: combine blended ranking, the incumbent
   buffer, the 15% core, and recovery-metered entry.

These are architectures, not a parameter search. The rank blend, buffer width,
replacement margin, core size, recovery windows, and entry fraction are frozen
before results. No candidate may be rescued by silently changing them.

## Causal construction

- Every price, return, volatility, moving-average, and rank input is available
  at the target-generation close.
- Incumbency and recovery state update only on the execution profile's scheduled
  decision dates.
- Exits and risk reductions are never delayed.
- The native strategy's total active-risk budget is preserved before the normal
  volatility target and drawdown control. The repair changes selector mix and
  transitions, not aggregate risk timing.
- Core and deferred-entry capital use SPY/RSP; no new BIL overlay is added.
- Backtest execution applies the configured signal lag after target generation.

## Validation contract

Every candidate is evaluated under:

- the configured W-WED, two-session causal path;
- Monday through Friday with one-session lag;
- W-WED one-, two-, and five-session lags plus daily/two-session execution;
- configured 5-basis-point and stressed 20-basis-point costs;
- full history, calendar years, named crises, and 1/3/5-year rolling windows;
- candidate-family CSCV/PBO;
- paired 63-session block-bootstrap deltas;
- concentration, holding identity, entry/exit count, replacement rate, turnover,
  and 2022 plus 2023-24 attribution;
- worst-drawdown attribution including peak/trough/recovery, asset loss
  contributors, turnover, exposure path, and missed SPY/QQQ recovery.

Schedule rows are dependence-aware diagnostics, not independent strategy trials.
Near-clone candidates do not multiply the evidentiary sample.

## Promotion-like screen

A non-reference candidate may be frozen for prospective monitoring only when:

1. configured-path CAGR is no worse than reference by more than 0.50 points;
2. configured-path maximum drawdown is no worse by more than 1.00 point;
3. median execution-profile CAGR is no worse by more than 0.50 points;
4. worst execution-profile drawdown improves by at least 2.00 points;
5. execution failure count falls;
6. average turnover does not rise by more than 25%;
7. median three-year rolling CAGR delta is positive;
8. at least 60% of calendar years have non-negative return delta;
9. at least 75% of named-crisis drawdown deltas are no worse than -1.50 points;
10. the focus CAGR delta remains non-negative at 20-basis-point costs;
11. family PBO is at most 25%;
12. the paired block-bootstrap probability of positive CAGR delta is at least
    70%, while drawdown damage worse than one point occurs in at most 35%.

Passing permits an untouched prospective shadow only. It does not authorize
automatic promotion or a live strategy/configuration change.

## Results

The initial fixed slate produced no retrospective pass. The configured
reference was 20.79% CAGR / -24.75% maximum drawdown; its median execution-path
CAGR was 19.54% and its worst execution-path drawdown was -29.87%.

`incumbent_buffer` was the only pre-registered mechanism to improve both the
configured and median-execution CAGR:

| Metric | Native reference | Incumbent buffer | Delta |
| --- | ---: | ---: | ---: |
| Configured CAGR | 20.79% | 20.91% | +0.13 pp |
| Configured maximum drawdown | -24.75% | -25.44% | -0.69 pp |
| Median execution CAGR | 19.54% | 19.80% | +0.25 pp |
| Worst execution drawdown | -29.87% | -30.00% | -0.12 pp |
| Aug. 2023-Jan. 2024 return | -12.30% | -11.12% | +1.18 pp |
| Median turnover | 0.0773 | 0.0703 | -9.1% |

The paired block bootstrap assigned only 65.3% probability to a positive CAGR
delta, below the fixed 70% gate. Only 40.9% of calendar years had non-negative
return deltas. The four broad era CAGR deltas were +0.16, -0.25, -0.57, and
positive 1.98 points. The mechanism therefore looks helpful in recent selector
churn, not stable enough to replace the native strategy.

The blended rank, diversified core, recovery meter, and initial integrated
candidate all reduced configured CAGR. The integrated candidate improved the
Aug. 2023-Jan. 2024 result to -8.87% and worst execution drawdown to -28.02%,
but configured CAGR fell to 19.20%. It is a risk/return trade rather than a
repair.

## Post-initial mechanism ablations

After the initial results were known, five combinations were run solely to
locate the source of the integrated failure. They are marked
`post_initial_mechanism_ablation`, cannot pass retrospectively, and were not
used to retune any parameter.

The strongest diagnostic combined the 63/126-day blended rank with the
incumbent buffer, omitting the core and recovery meter:

- configured CAGR 20.36%, 0.42 points below reference;
- median execution CAGR 20.13%, 0.59 points above reference;
- worst execution drawdown -28.86%, 1.01 points better than reference;
- Aug. 2023-Jan. 2024 return -9.09%, 3.20 points better than reference;
- 32.1% block-bootstrap probability of a positive full-history CAGR delta;
- only 40.9% of calendar years with a non-negative return delta.

That is the best execution-robustness trade in this study, but it does not
establish a better strategy. Core and recovery-meter combinations remained
return-dilutive. The evidence supports prospective monitoring of selector
churn and incumbent retention, not a live configuration change.

## Cleanup and retained evidence

This batch is closed as `no_candidate_passed`, with the explicit decision
`retain_native_reference`.

- None of the candidates is registered in runtime configuration, monitoring,
  candidate curation, or a dashboard leaderboard.
- The static 15% core, recovery-metered entry, and initial integrated
  replacement hypotheses are closed. They should not be rerun during routine
  daily updates or revived through small parameter changes.
- The post-initial combinations remain labeled diagnostic and cannot become
  promotion evidence on this history.
- CSVs, the manifest, and the research runner are retained as the audit trail;
  failed evidence is archived rather than deleted.
- The causal incumbent-buffer primitive and drawdown-attribution utility remain
  because they are reusable diagnostics and are covered by tests. Reopening the
  strategy question requires genuinely new forward data or a materially
  different selector mechanism.

Reproduce the manifested artifact set with:

```bash
poetry run trade-bot run-i111-selector-transition-repair \
  --config configs/baseline.yaml
```
