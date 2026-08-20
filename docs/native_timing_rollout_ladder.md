# Native Timing Rollout Ladder

Status: pre-registered implementation-safety research. Allocation authority is
zero.

The progression and tail-guard studies did not add positive i111-family utility
above the frozen mild curve. This study therefore does not add another timing
signal. It tests fixed fractions of the already-frozen mild correction:

- `mild_scale_25`: apply 25% of the native-to-mild defense change;
- `mild_scale_50`: apply 50%;
- `mild_scale_75`: apply 75%;
- `mild_frozen`: apply 100%.

The fractions are a rollout-risk ladder, not parameter optimization. A
prospective starting shadow is considered implementation-safe only if it:

1. improves focus CAGR versus native in all four outer folds;
2. improves i111-family median CAGR versus native;
3. keeps focus maximum-drawdown damage within 0.50 points of native;
4. keeps at least 90% of strategy/crisis checks within one drawdown point of
   native;
5. remains positive versus native at 20 basis points;
6. changes current focus defense by no more than five percentage points;
7. is monotonic between native and the full mild curve in allocation and CAGR.

Passing identifies a conservative prospective shadow size only. It does not
authorize a live allocation change on history already used to define the mild
curve.
