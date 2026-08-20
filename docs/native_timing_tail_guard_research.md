# Native Timing Tail-Guard Research

Status: pre-registered retrospective third-wave research. Allocation authority
is zero.

## Motivation

The conditional warning-progression model did not beat its warning-episode base
rate. Broad confirmation restoration improved drawdown but surrendered too much
of the frozen mild curve's CAGR gain. This wave therefore abandons learned
hazard ranking and asks a narrower question: can the mild curve retain native
protection only after severe deterioration is already observable?

## Frozen benchmark and fixed roster

The benchmark remains `mild_frozen` with its previously frozen parameters and
one-session execution lag. Four architectures are declared:

1. `three_group_native_veto`: never remain below native defense when at least
   three independent break groups are active.
2. `two_group_drawdown8_veto`: apply that veto only when at least two break
   groups are active and the worse SPY/QQQ trailing drawdown is at least 8%.
3. `two_group_volaccel_veto`: apply it only when at least two break groups are
   active and 21-day realized volatility exceeds its 63-day estimate by at
   least five annualized volatility points.
4. `drawdown_5_10_native_ramp`: progressively restore the gap from mild to
   native defense as the worse SPY/QQQ drawdown moves from 5% to 10%.

No additional threshold or architecture may be added after results are viewed
in this wave.

## Gate

The same ten gates from the conditional progression study apply, except no
forecast-Brier gate is required because these are observable-state safeguards,
not forecasting models. A survivor must retain at least 75% of the mild curve's
focus CAGR gain, improve focus drawdown versus mild, keep at least three of four
focus folds positive versus native, improve i111-family median utility versus
mild, remain robust at 20 basis points, and satisfy the native/crisis drawdown
budgets.

Even a full retrospective pass permits only a newly frozen prospective shadow.
It cannot authorize live sizing on already inspected history.
