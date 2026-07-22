# Risk-Timing Research

Status: implemented as a research-visible, allocation-disabled layer. Last reviewed: 2026-07-21.

## Decision

The former quantitative risk-status multiplier is no longer allowed to size the
operating portfolio. The broad `risk_score` and `green/yellow/orange/red` label
remain useful fragility diagnostics, but their historical allocation rule was
too early in intact markets and did not improve the native strategy's overall
risk-adjusted outcome enough to justify authority.

The replacement timing model is point-in-time and uses four explicit market
groups:

- credit: bearish HYG/LQD state plus a material negative HYG one-month return;
- volatility: risk-off VIXY state plus a material positive VIXY one-month return;
- breadth: bearish RSP/SPY state plus meaningful equal-weight underperformance;
- trend: negative short/intermediate SPY or QQQ trend, or both slow trend states bearish.

It reports `normal`, `watch`, `fragile_intact`, `warning`,
`confirmed_break`, `severe_break`, or `stabilizing`. Fragility and stabilization
are informational. A raw budget reduction begins at warning; meaningful defense
requires two independent break groups; the severe state also requires a
double-digit SPY/QQQ drawdown. Broad short-horizon recovery can override a stale
slow risk score.

## Empirical gate

The point-in-time replay used 1,020 W-WED origins from the current cached price
history. The non-overlapping weekly comparison was:

| Policy | CAGR | Max drawdown | Delta CAGR vs native | Drawdown change vs native |
| --- | ---: | ---: | ---: | ---: |
| Native strategy | 18.74% | -26.21% | 0.00 pp | 0.00 pp |
| Legacy risk-status sizing | 13.55% | -17.76% | -5.19 pp | +8.45 pp |
| Confirmation-timed candidate | 18.12% | -25.36% | -0.62 pp | +0.85 pp |
| Candidate plus portfolio limits | 13.69% | -18.63% | -5.05 pp | +7.59 pp |

The candidate removed most of the legacy return drag, but bought only 0.85
percentage points of maximum-drawdown improvement at a 0.62-point CAGR cost;
its Calmar ratio was essentially unchanged. Era checks also failed to show
stable downside benefit:
confirmed-only variants cost return without changing maximum drawdown in
2007-2015 and produced only a few basis points of return improvement, again
without changing maximum drawdown, in 2016-2026.

This is insufficient evidence for live sizing. The active configuration therefore sets:

```yaml
risk_timing_sizing_authority: 0.0
risk_timing_calibration_status: insufficient
risk_timing_policy_version: confirmed_v1
```

Configuration validation fails closed: nonzero authority is rejected unless
calibration status is `provisional` or `validated`.

## Interpretation

The timing layer is now a falsifiable research claim, not a hidden source of
cash. A high fragility score can warn that conditions are vulnerable, but it
cannot independently reduce target risk. Today's final allocation can still be
defensive because the native strategy is defensive, an accepted quantitative
macro category acts, or hard portfolio constraints bind; the causal attribution
must identify which layer actually changed weights.

Future promotion requires a predeclared candidate and thresholds, train/test or
walk-forward evidence, meaningful drawdown improvement, tolerable CAGR cost,
and stability across crisis families and execution schedules. Historical stage
labels based on known future break dates remain diagnostic only and may not enter
the rule.
