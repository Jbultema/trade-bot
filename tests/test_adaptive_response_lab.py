from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trade_bot.config import StrategyConfig
from trade_bot.research.adaptive_response_lab import (
    ALTERNATIVES,
    Context,
    Features,
    Policy,
    build_policy_targets,
    combination_policies,
    controls,
    downside_cusum,
    execute,
    followups,
    initial_policies,
    recovery_exposure,
    schedule_targets,
    select_winners,
)


def _context() -> Context:
    index = pd.bdate_range("2008-01-01", periods=500)
    rng = np.random.default_rng(91)
    common = rng.normal(0.0006, 0.01, len(index))
    common[380:400] -= 0.025
    prices = pd.DataFrame(index=index)
    for ticker in ["QQQ", "SPY", *ALTERNATIVES, "XLK", "XLY"]:
        changes = common + rng.normal(0.0001, 0.003, len(index))
        prices[ticker] = 100 * np.cumprod(1 + changes)
    prices["CASH"] = 100 * np.cumprod(np.full(len(index), 1.0001))
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["QQQ", "XLK", "XLY"],
        lookback_days=63,
        skip_days=5,
        top_n=2,
        defensive_ticker="CASH",
        min_return=0.0,
        ranking_metric="risk_adjusted_return",
        weighting="risk_adjusted_score",
        max_asset_weight=0.35,
    )
    return Context("test", prices, strategy, "QQQ", index[252], "test", "test")


@pytest.mark.parametrize("policy", initial_policies(), ids=lambda p: p.name)
def test_targets_are_causal_and_respect_cash_and_asset_limits(policy: Policy) -> None:
    context = _context()
    full, _ = build_policy_targets(Features(context), policy)
    truncated = replace(context, prices=context.prices.iloc[:423])
    prefix, _ = build_policy_targets(Features(truncated), policy)

    pd.testing.assert_frame_equal(full.iloc[:423], prefix)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-12)
    assert full.min().min() >= 0
    assert full.drop(columns="CASH").max().max() <= 0.3500001
    assert full.loc[full.index < context.evaluation_start, "CASH"].eq(1).all()


def test_static_risk_control_scales_actual_executed_risk() -> None:
    context = _context()
    target, _ = build_policy_targets(Features(context), controls()[0])
    reference = execute(context, target)
    scaled = execute(context, target, risk_scale=0.75)

    pd.testing.assert_frame_equal(
        reference.weights.drop(columns="CASH") * 0.75,
        scaled.weights.drop(columns="CASH"),
    )
    np.testing.assert_allclose(scaled.weights.sum(axis=1), 1.0)


def test_buy_hold_benchmark_is_not_subject_to_strategy_asset_cap() -> None:
    context = _context()
    target, _ = build_policy_targets(
        Features(context), Policy("spy", "benchmark", selection="buy_spy", cadence="daily")
    )
    assert target["SPY"].iloc[-1] == 1.0


def test_cash_returns_are_credited_without_double_execution_lag() -> None:
    context = _context()
    target, _ = build_policy_targets(
        Features(context), Policy("cash", "benchmark", selection="cash", cadence="daily")
    )
    result = execute(context, target, controlled=False, cost=0)
    np.testing.assert_allclose(result.returns.iloc[1:], 0.0001, atol=1e-12)


def test_cusum_detects_negative_surprise_without_future_observations() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0002, 0.002, 120)
    returns[-3:] = -0.04
    prices = pd.Series(100 * np.cumprod(1 + returns))
    score = downside_cusum(prices)
    assert score.iloc[-1] > score.iloc[-4] + 10
    pd.testing.assert_series_equal(score.iloc[:110], downside_cusum(prices.iloc[:110]))


def test_recovery_requires_repair_and_restores_exposure_incrementally() -> None:
    stress = pd.Series([0.0, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1])
    repair = pd.Series([False, False, False, False, True, True, True])
    policy = Policy(
        "repair", "recovery", exit_days=1, release_days=2, reduction=0.6, reentry_step=0.2
    )
    exposure = recovery_exposure(stress, repair, policy)
    np.testing.assert_allclose(exposure, [1, 0.4, 0.4, 0.4, 0.4, 0.6, 0.8])


def test_weekly_schedule_does_not_treat_partial_week_as_a_signal() -> None:
    dates = pd.to_datetime(["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23"])
    weights = pd.DataFrame({"QQQ": [0.2, 0.4, 0.8, 0.9], "CASH": [0.8, 0.6, 0.2, 0.1]}, index=dates)
    scheduled = schedule_targets(weights, pd.Series(dates.weekday == 2, index=dates))
    assert scheduled["QQQ"].eq(0.2).all()


def test_winner_selection_cannot_read_later_diagnostic_results() -> None:
    rows = []
    for name, family, training, later in (
        ("slow_reference", "control", 0.10, 0.10),
        ("stable", "adaptive_blend", 0.12, -0.10),
        ("future_lucky", "adaptive_blend", 0.05, 1.0),
    ):
        for period in ["development", "validation", "historical_test", "recent_diagnostic"]:
            cagr = training if period in ["development", "validation"] else later
            rows.append(
                {
                    "context": "test",
                    "candidate": name,
                    "family": family,
                    "period": period,
                    "cagr": cagr,
                    "sharpe": cagr * 5,
                    "max_drawdown": -0.2,
                    "annual_turnover": 5,
                }
            )
    assert select_winners(pd.DataFrame(rows)).iloc[0].candidate == "stable"


def test_two_followups_exist_for_every_family_and_combinations_keep_lineage() -> None:
    initial = initial_policies()
    assert len(initial) == 20
    parents = {policy.family: policy for policy in initial}
    for parent in parents.values():
        second = followups(parent, 2)
        third = followups(second[0], 3)
        assert len(second) == len(third) == 3
        assert all(policy.parent == parent.name for policy in second)
        assert all(policy.parent == second[0].name for policy in third)
    combinations = combination_policies(parents)
    assert len(combinations) == 6
    assert all(policy.parent and policy.round == 4 for policy in combinations)


def test_newly_listed_asset_cannot_enter_before_its_lookback_exists() -> None:
    context = _context()
    context.prices.loc[context.prices.index[:350], "XLK"] = np.nan
    policy = Policy(
        "adaptive", "adaptive_blend", selection="adaptive", fast_days=21, cadence="adaptive"
    )
    targets, _ = build_policy_targets(Features(context), policy)
    assert targets["XLK"].iloc[:376].eq(0).all()


def test_disabling_daily_emergency_rebalances_is_a_real_cadence_ablation() -> None:
    context = _context()
    policy = Policy("protect", "fast_protection", protection="continuous")
    daily, _ = build_policy_targets(Features(context), policy)
    weekly, _ = build_policy_targets(Features(context), replace(policy, emergency_daily=False))
    changes = weekly.diff().abs().sum(axis=1).gt(1e-8)
    assert (weekly.index[changes].weekday == 2).all()
    assert (daily - weekly).abs().sum().sum() > 0
