from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import BotConfig
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.defensive_bias_calibration import (
    CRISIS_WINDOWS,
    _result_from_execution_weights,
    _return_slice_stats,
    comparable_strategy_family,
)
from trade_bot.research.defensive_correction_search import (
    effective_defensive_weight_path,
)
from trade_bot.research.native_timing_hazard import (
    OUTER_FOLDS,
    _add_non_deterioration_age,
    _fit_probability_model,
    apply_continuous_defense_budget,
    build_family_state_panel,
    build_market_hazard_panel,
    continuous_defense_target,
)
from trade_bot.research.strategy_outcome_utility import terminal_wealth_from_cagr

DEFAULT_NATIVE_TIMING_PROGRESSION_DIR = Path("reports/native_timing_progression")
WARNING_DEFENSE_FLOOR = 0.25
WARNING_DEFENSE_RISE = 0.10
WARNING_MODEL_C = 0.20
EXTRA_EXECUTION_LAG = 1
MILD_PARAMETERS = {
    "defense_floor": 0.00,
    "hazard_slope": 0.60,
    "defense_ceiling": 0.65,
    "native_blend": 0.75,
    "break_acceleration": 0.0,
    "age_decay": 0.0,
}
POLICY_ROSTER = (
    "mild_scale_25",
    "mild_scale_50",
    "mild_scale_75",
    "mild_frozen",
    "progression_restore_10",
    "progression_restore_20",
    "confirmation_restore_10",
    "hybrid_restore_15",
    "native_severe_veto",
    "three_group_native_veto",
    "two_group_drawdown8_veto",
    "two_group_volaccel_veto",
    "drawdown_5_10_native_ramp",
)
PROGRESSION_FEATURES = (
    "break_count",
    "spy_drawdown_252",
    "qqq_drawdown_252",
    "breadth_return_63",
    "credit_return_63",
    "realized_vol_21",
    "realized_vol_acceleration",
    "family_defense",
    "family_defense_delta_21",
    "family_drawdown",
    "warning_age_weeks",
    "non_deterioration_age_weeks",
)


@dataclass(frozen=True)
class NativeTimingProgressionRun:
    warning_panel: pd.DataFrame
    oos_predictions: pd.DataFrame
    model_diagnostics: pd.DataFrame
    policy_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    crisis_metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    current_read: pd.DataFrame
    promotion_gates: pd.DataFrame
    rollout_ladder: pd.DataFrame
    output_paths: dict[str, Path]


def build_warning_progression_panel(
    baseline_run: BaselineRun,
) -> tuple[pd.DataFrame, dict[str, str]]:
    strategy_families = {
        name: family
        for name in baseline_run.results
        if (family := comparable_strategy_family(name)) is not None
    }
    prices = baseline_run.prices.sort_index()
    market = build_market_hazard_panel(prices)
    family = build_family_state_panel(
        baseline_run,
        strategy_families,
        market["origin_date"],
    )
    panel = family.merge(
        market,
        on="origin_date",
        how="left",
        validate="many_to_one",
    )
    panel = _add_non_deterioration_age(panel)
    panel["warning_eligible"] = (
        pd.to_numeric(panel["family_defense"], errors="coerce").ge(WARNING_DEFENSE_FLOOR)
        | pd.to_numeric(panel["family_defense_delta_21"], errors="coerce").ge(WARNING_DEFENSE_RISE)
        | pd.to_numeric(panel["break_count"], errors="coerce").ge(1.0)
    )
    return add_warning_episode_weights(panel), strategy_families


def add_warning_episode_weights(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["family", "origin_date"]).reset_index(drop=True)
    episode_parts: list[pd.Series] = []
    for family_name, group in panel.groupby("family", sort=False):
        eligible = group["warning_eligible"].fillna(False).astype(bool)
        starts = eligible & ~eligible.shift(fill_value=False)
        episode_number = starts.cumsum()
        episode = pd.Series("", index=group.index, dtype=object)
        episode.loc[eligible] = family_name + ":warning:" + episode_number.loc[eligible].astype(str)
        episode_parts.append(episode)
    panel["warning_episode"] = pd.concat(episode_parts).sort_index()
    panel["warning_episode_weight"] = 0.0
    eligible = panel["warning_eligible"] & panel["forward_break"].notna()
    counts = panel.loc[eligible].groupby("warning_episode")["warning_episode"].transform("size")
    panel.loc[eligible, "warning_episode_weight"] = 1.0 / counts
    return panel


def build_conditional_oos_predictions(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matured = panel[panel["forward_break"].notna()].copy()
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for fold_name, start_text, end_text in OUTER_FOLDS:
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        all_train = matured[pd.to_datetime(matured["maturity_date"]).lt(start)].copy()
        train = matured[
            pd.to_datetime(matured["maturity_date"]).lt(start) & matured["warning_eligible"]
        ].copy()
        test = matured[pd.to_datetime(matured["origin_date"]).between(start, end)].copy()
        if train.empty or test.empty:
            continue
        train["episode_weight"] = train["warning_episode_weight"]
        base_rate = float(
            np.average(
                pd.to_numeric(train["forward_break"], errors="coerce"),
                weights=pd.to_numeric(train["warning_episode_weight"], errors="coerce").fillna(0.0),
            )
        )
        unconditional_base_rate = float(
            np.average(
                pd.to_numeric(all_train["forward_break"], errors="coerce"),
                weights=pd.to_numeric(all_train["episode_weight"], errors="coerce").fillna(1.0),
            )
        )
        probability = _fit_probability_model(
            train,
            test,
            PROGRESSION_FEATURES,
            c_value=WARNING_MODEL_C,
        )
        output = test[
            [
                "origin_date",
                "maturity_date",
                "family",
                "forward_break",
                "forward_worst_drawdown",
                "warning_episode",
                "warning_episode_weight",
                "warning_eligible",
                "break_count",
                "warning_age_weeks",
                "non_deterioration_age_weeks",
                "spy_drawdown_252",
                "qqq_drawdown_252",
                "realized_vol_21",
                "realized_vol_acceleration",
            ]
        ].copy()
        output["outer_fold"] = fold_name
        output["training_realized_vol_q75"] = float(
            pd.to_numeric(all_train["realized_vol_21"], errors="coerce").quantile(0.75)
        )
        output["mild_base_rate_probability"] = unconditional_base_rate
        output["warning_base_rate_probability"] = base_rate
        output["warning_progression_probability"] = probability
        denominator = max(1.0 - base_rate, 1e-9)
        output["warning_excess_probability"] = (
            (output["warning_progression_probability"] - base_rate) / denominator
        ).clip(0.0, 1.0)
        output.loc[~output["warning_eligible"], "warning_excess_probability"] = 0.0
        predictions.append(output)

        eligible_test = output[output["warning_eligible"]].copy()
        if eligible_test.empty:
            continue
        weight = pd.to_numeric(eligible_test["warning_episode_weight"], errors="coerce").fillna(0.0)
        label = pd.to_numeric(eligible_test["forward_break"], errors="coerce")
        model_probability = pd.to_numeric(
            eligible_test["warning_progression_probability"], errors="coerce"
        )
        model_brier = float(np.average((label - model_probability) ** 2, weights=weight))
        base_brier = float(np.average((label - base_rate) ** 2, weights=weight))
        diagnostics.append(
            {
                "outer_fold": fold_name,
                "train_warning_rows": len(train),
                "train_warning_episodes": train["warning_episode"].nunique(),
                "test_warning_rows": len(eligible_test),
                "test_warning_episodes": eligible_test["warning_episode"].nunique(),
                "warning_base_rate": base_rate,
                "conditional_brier_score": model_brier,
                "warning_base_rate_brier_score": base_brier,
                "brier_improvement_vs_warning_base_rate": base_brier - model_brier,
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(diagnostics)


def _policy_target(
    *,
    policy: str,
    base_defense: pd.Series,
    probability: pd.Series,
    excess_probability: pd.Series,
    break_count: pd.Series,
    non_deterioration_age: pd.Series,
    market_drawdown: pd.Series | None = None,
    realized_vol_acceleration: pd.Series | None = None,
) -> pd.Series:
    mild = continuous_defense_target(
        probability,
        base_defense,
        break_count,
        non_deterioration_age,
        policy="constant_mild_continuous_existing",
        parameters=MILD_PARAMETERS,
    )
    confirmation_score = ((break_count - 1.0) / 2.0).clip(0.0, 1.0)
    market_drawdown = (
        market_drawdown if market_drawdown is not None else pd.Series(0.0, index=base_defense.index)
    )
    realized_vol_acceleration = (
        realized_vol_acceleration
        if realized_vol_acceleration is not None
        else pd.Series(0.0, index=base_defense.index)
    )
    native_floor = pd.concat([mild, base_defense], axis=1).max(axis=1)
    if policy.startswith("mild_scale_"):
        scale = float(policy.rsplit("_", maxsplit=1)[-1]) / 100.0
        target = base_defense + scale * (mild - base_defense)
    elif policy == "mild_frozen":
        target = mild
    elif policy == "progression_restore_10":
        target = mild + 0.10 * excess_probability
    elif policy == "progression_restore_20":
        target = mild + 0.20 * excess_probability
    elif policy == "confirmation_restore_10":
        target = mild + 0.10 * confirmation_score
    elif policy == "hybrid_restore_15":
        restoration = pd.concat(
            [0.15 * excess_probability, 0.15 * confirmation_score],
            axis=1,
        ).max(axis=1)
        target = mild + restoration
    elif policy == "native_severe_veto":
        target = mild.where(break_count.lt(2.0), native_floor)
    elif policy == "three_group_native_veto":
        target = mild.where(break_count.lt(3.0), native_floor)
    elif policy == "two_group_drawdown8_veto":
        severe = break_count.ge(2.0) & market_drawdown.le(-0.08)
        target = mild.where(~severe, native_floor)
    elif policy == "two_group_volaccel_veto":
        severe = break_count.ge(2.0) & realized_vol_acceleration.ge(0.05)
        target = mild.where(~severe, native_floor)
    elif policy == "drawdown_5_10_native_ramp":
        severity = ((-market_drawdown - 0.05) / 0.05).clip(0.0, 1.0)
        target = mild + severity * (native_floor - mild)
    else:
        raise ValueError(f"Unknown progression policy: {policy}")
    return target.clip(0.0, 1.0)


def build_progression_policy_results(
    baseline_run: BaselineRun,
    config: BotConfig,
    strategy_families: dict[str, str],
    predictions: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str], BacktestResult],
    dict[tuple[str, str], pd.DataFrame],
]:
    prices = baseline_run.prices.sort_index()
    strategies = [strategy for strategy, family in strategy_families.items() if family == "i111"]
    weights = {
        (strategy, policy): baseline_run.results[strategy].weights.copy()
        for strategy in strategies
        for policy in POLICY_ROSTER
    }
    for fold_name, start_text, end_text in OUTER_FOLDS:
        fold = predictions[predictions["outer_fold"].eq(fold_name)].copy()
        fold = fold[fold["family"].eq("i111")]
        if fold.empty:
            continue
        start = max(pd.Timestamp(start_text), pd.Timestamp(fold["origin_date"].min()))
        end = min(pd.Timestamp(end_text), pd.Timestamp(fold["origin_date"].max()))
        index = prices.index[prices.index.to_series().between(start, end).to_numpy()]
        weekly = fold.set_index("origin_date")[
            [
                "mild_base_rate_probability",
                "warning_excess_probability",
                "break_count",
                "non_deterioration_age_weeks",
                "spy_drawdown_252",
                "qqq_drawdown_252",
                "realized_vol_acceleration",
            ]
        ].sort_index()
        state = weekly.reindex(index).ffill()
        market_drawdown = state[["spy_drawdown_252", "qqq_drawdown_252"]].min(axis=1)
        for strategy in strategies:
            base = baseline_run.results[strategy]
            base_weights = base.weights.loc[index]
            base_defense = effective_defensive_weight_path(base_weights)
            for policy in POLICY_ROSTER:
                target = _policy_target(
                    policy=policy,
                    base_defense=base_defense,
                    probability=state["mild_base_rate_probability"],
                    excess_probability=state["warning_excess_probability"],
                    break_count=state["break_count"],
                    non_deterioration_age=state["non_deterioration_age_weeks"],
                    market_drawdown=market_drawdown,
                    realized_vol_acceleration=state["realized_vol_acceleration"],
                )
                target = target.shift(EXTRA_EXECUTION_LAG).fillna(base_defense)
                adjusted = apply_continuous_defense_budget(
                    base_weights,
                    target,
                    spy_bridge=False,
                )
                destination = weights[(strategy, policy)]
                for column in adjusted.columns.difference(destination.columns):
                    destination[column] = 0.0
                destination.loc[index, adjusted.columns] = adjusted
    results = {
        key: _result_from_execution_weights(
            baseline_run.results[key[0]],
            prices,
            policy_weights,
            transaction_cost_bps=float(config.execution.transaction_cost_bps),
            name=f"{key[0]}__{key[1]}",
        )
        for key, policy_weights in weights.items()
    }
    return results, weights


def evaluate_progression_policies(
    baseline_run: BaselineRun,
    config: BotConfig,
    strategy_families: dict[str, str],
    predictions: pd.DataFrame,
    results: dict[tuple[str, str], BacktestResult],
    weights: dict[tuple[str, str], pd.DataFrame],
    *,
    policy_roster: tuple[str, ...] = POLICY_ROSTER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(predictions["origin_date"].min())
    end = pd.Timestamp(predictions["origin_date"].max())
    strategy_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    crisis_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    strategies = [strategy for strategy, family in strategy_families.items() if family == "i111"]
    for strategy in strategies:
        base = baseline_run.results[strategy]
        mild = results[(strategy, "mild_frozen")]
        base_stats = _return_slice_stats(base.returns, start, end)
        mild_stats = _return_slice_stats(mild.returns, start, end)
        base_cagr = float(base_stats["annualized_return"])
        base_dd = float(base_stats["max_drawdown"])
        mild_cagr = float(mild_stats["annualized_return"])
        mild_dd = float(mild_stats["max_drawdown"])
        for policy in policy_roster:
            candidate = results[(strategy, policy)]
            stats = _return_slice_stats(candidate.returns, start, end)
            cagr = float(stats["annualized_return"])
            drawdown = float(stats["max_drawdown"])
            wealth = float(
                terminal_wealth_from_cagr(
                    cagr,
                    years=15,
                    starting_account_value=220_000.0,
                    annual_contribution=4_000.0,
                    contribution_timing="monthly",
                ).iloc[0]
            )
            defense = effective_defensive_weight_path(candidate.weights)
            base_defense = effective_defensive_weight_path(base.weights)
            delta = (defense - base_defense).abs().loc[start:end]
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "family": "i111",
                    "policy": policy,
                    "candidate_cagr": cagr,
                    "candidate_max_drawdown": drawdown,
                    "cagr_delta_vs_native": cagr - base_cagr,
                    "max_drawdown_delta_vs_native": drawdown - base_dd,
                    "cagr_delta_vs_mild": cagr - mild_cagr,
                    "max_drawdown_delta_vs_mild": drawdown - mild_dd,
                    "utility_delta_vs_mild": (cagr - mild_cagr) + 0.75 * (drawdown - mild_dd),
                    "wealth_15y": wealth,
                    "active_day_rate": float(delta.gt(0.025).mean()),
                    "mean_absolute_defense_change": float(delta.mean()),
                }
            )
            for fold_name, fold_start, fold_end in OUTER_FOLDS:
                native_fold = _return_slice_stats(base.returns, fold_start, fold_end)
                mild_fold = _return_slice_stats(mild.returns, fold_start, fold_end)
                candidate_fold = _return_slice_stats(candidate.returns, fold_start, fold_end)
                fold_rows.append(
                    {
                        "strategy": strategy,
                        "policy": policy,
                        "outer_fold": fold_name,
                        "cagr_delta_vs_native": float(candidate_fold["annualized_return"])
                        - float(native_fold["annualized_return"]),
                        "max_drawdown_delta_vs_native": float(candidate_fold["max_drawdown"])
                        - float(native_fold["max_drawdown"]),
                        "cagr_delta_vs_mild": float(candidate_fold["annualized_return"])
                        - float(mild_fold["annualized_return"]),
                        "max_drawdown_delta_vs_mild": float(candidate_fold["max_drawdown"])
                        - float(mild_fold["max_drawdown"]),
                    }
                )
            for crisis, crisis_start, crisis_end in CRISIS_WINDOWS:
                native_crisis = _return_slice_stats(base.returns, crisis_start, crisis_end)
                candidate_crisis = _return_slice_stats(candidate.returns, crisis_start, crisis_end)
                crisis_rows.append(
                    {
                        "strategy": strategy,
                        "policy": policy,
                        "crisis": crisis,
                        "max_drawdown_delta_vs_native": float(candidate_crisis["max_drawdown"])
                        - float(native_crisis["max_drawdown"]),
                    }
                )
            for cost_bps in (10.0, 20.0):
                base_cost_result = _result_from_execution_weights(
                    base,
                    baseline_run.prices,
                    base.weights,
                    transaction_cost_bps=cost_bps,
                    name=f"{strategy}__native__{cost_bps:g}bps",
                )
                cost_result = _result_from_execution_weights(
                    base,
                    baseline_run.prices,
                    weights[(strategy, policy)],
                    transaction_cost_bps=cost_bps,
                    name=f"{strategy}__{policy}__{cost_bps:g}bps",
                )
                base_cost_stats = _return_slice_stats(base_cost_result.returns, start, end)
                cost_stats = _return_slice_stats(cost_result.returns, start, end)
                cost_rows.append(
                    {
                        "strategy": strategy,
                        "policy": policy,
                        "cost_bps": cost_bps,
                        "cagr_delta_vs_native": float(cost_stats["annualized_return"])
                        - float(base_cost_stats["annualized_return"]),
                    }
                )
    return (
        pd.DataFrame(strategy_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(crisis_rows),
        pd.DataFrame(cost_rows),
    )


def build_current_progression_read(
    baseline_run: BaselineRun,
    config: BotConfig,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    matured = panel[panel["forward_break"].notna() & panel["warning_eligible"]].copy()
    all_matured = panel[panel["forward_break"].notna()].copy()
    matured["episode_weight"] = matured["warning_episode_weight"]
    latest = panel.sort_values("origin_date").groupby("family", as_index=False).tail(1)
    probability = _fit_probability_model(
        matured,
        latest,
        PROGRESSION_FEATURES,
        c_value=WARNING_MODEL_C,
    )
    rows: list[dict[str, object]] = []
    for position, (_, state) in enumerate(latest.iterrows()):
        family = str(state["family"])
        train_family = matured[matured["family"].eq(family)]
        base_rate = float(
            np.average(
                train_family["forward_break"],
                weights=train_family["warning_episode_weight"],
            )
        )
        mild_base_rate = float(
            np.average(
                all_matured["forward_break"],
                weights=all_matured["episode_weight"],
            )
        )
        excess = (
            max(0.0, (float(probability[position]) - base_rate) / (1.0 - base_rate))
            if bool(state["warning_eligible"])
            else 0.0
        )
        strategies = [
            strategy
            for strategy in baseline_run.results
            if comparable_strategy_family(strategy) == family
        ]
        for strategy in strategies:
            base_weights = baseline_run.results[strategy].weights.tail(1)
            base_defense = effective_defensive_weight_path(base_weights)
            for policy in POLICY_ROSTER:
                target = _policy_target(
                    policy=policy,
                    base_defense=base_defense,
                    probability=pd.Series([mild_base_rate], index=base_weights.index),
                    excess_probability=pd.Series([excess], index=base_weights.index),
                    break_count=pd.Series([float(state["break_count"])], index=base_weights.index),
                    non_deterioration_age=pd.Series(
                        [float(state["non_deterioration_age_weeks"])],
                        index=base_weights.index,
                    ),
                    market_drawdown=pd.Series(
                        [
                            min(
                                float(state["spy_drawdown_252"]),
                                float(state["qqq_drawdown_252"]),
                            )
                        ],
                        index=base_weights.index,
                    ),
                    realized_vol_acceleration=pd.Series(
                        [float(state["realized_vol_acceleration"])],
                        index=base_weights.index,
                    ),
                )
                rows.append(
                    {
                        "market_date": base_weights.index[-1],
                        "strategy": strategy,
                        "family": family,
                        "policy": policy,
                        "warning_eligible": bool(state["warning_eligible"]),
                        "mild_base_rate": mild_base_rate,
                        "warning_base_rate": base_rate,
                        "warning_progression_probability": float(probability[position]),
                        "warning_excess_probability": excess,
                        "break_count": float(state["break_count"]),
                        "base_defensive_weight": float(base_defense.iloc[0]),
                        "target_defensive_weight": float(target.iloc[0]),
                        "defensive_weight_change": float(target.iloc[0] - base_defense.iloc[0]),
                        "allocation_authority": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def build_progression_gates(
    config: BotConfig,
    diagnostics: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    crises: pd.DataFrame,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    focus_name = config.primary_strategy
    model_beats_base = float(diagnostics["brier_improvement_vs_warning_base_rate"].mean()) > 0.0
    focus_mild = metrics[
        metrics["strategy"].eq(focus_name) & metrics["policy"].eq("mild_frozen")
    ].iloc[0]
    mild_gain = float(focus_mild["cagr_delta_vs_native"])
    for policy in POLICY_ROSTER:
        focus = metrics[metrics["strategy"].eq(focus_name) & metrics["policy"].eq(policy)].iloc[0]
        family = metrics[metrics["policy"].eq(policy)]
        focus_folds = folds[folds["strategy"].eq(focus_name) & folds["policy"].eq(policy)]
        policy_crises = crises[crises["policy"].eq(policy)]
        focus_cost = costs[
            costs["strategy"].eq(focus_name)
            & costs["policy"].eq(policy)
            & costs["cost_bps"].eq(20.0)
        ]
        gates = {
            "conditional_model_beats_warning_base": (
                model_beats_base if "progression" in policy or "hybrid" in policy else True
            ),
            "retain_75pct_mild_cagr_gain": float(focus["cagr_delta_vs_native"]) >= 0.75 * mild_gain,
            "drawdown_improves_vs_mild": float(focus["max_drawdown_delta_vs_mild"]) > 0.0,
            "native_drawdown_budget": float(focus["max_drawdown_delta_vs_native"]) >= -0.01
            and float(focus["candidate_max_drawdown"]) >= -0.30,
            "three_of_four_folds_positive_vs_native": int(
                focus_folds["cagr_delta_vs_native"].gt(0.0).sum()
            )
            >= 3,
            "three_of_four_folds_nonworse_dd_vs_mild": int(
                focus_folds["max_drawdown_delta_vs_mild"].ge(0.0).sum()
            )
            >= 3,
            "family_utility_improves_vs_mild": float(family["utility_delta_vs_mild"].median())
            > 0.0,
            "crisis_drawdown_budget": float(
                policy_crises["max_drawdown_delta_vs_native"].ge(-0.015).mean()
            )
            >= 0.75,
            "positive_at_20bps": bool(
                not focus_cost.empty and float(focus_cost.iloc[0]["cagr_delta_vs_native"]) > 0.0
            ),
            "material_allocation_effect": float(focus["active_day_rate"]) >= 0.05
            and float(focus["mean_absolute_defense_change"]) >= 0.025,
        }
        failed = [name for name, passed in gates.items() if not passed]
        rows.append(
            {
                "policy": policy,
                "focus_cagr_delta_vs_native": focus["cagr_delta_vs_native"],
                "focus_cagr_delta_vs_mild": focus["cagr_delta_vs_mild"],
                "focus_max_drawdown_delta_vs_native": focus["max_drawdown_delta_vs_native"],
                "focus_max_drawdown_delta_vs_mild": focus["max_drawdown_delta_vs_mild"],
                **gates,
                "gate_pass_count": sum(gates.values()),
                "failed_gates": ", ".join(failed),
                "retrospective_gate_passed": all(gates.values()),
                "automatic_promotion_allowed": False,
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["retrospective_gate_passed", "gate_pass_count", "focus_cagr_delta_vs_native"],
        ascending=[False, False, False],
    )


def build_rollout_ladder(
    config: BotConfig,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    crises: pd.DataFrame,
    costs: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    policies = (
        "mild_scale_25",
        "mild_scale_50",
        "mild_scale_75",
        "mild_frozen",
    )
    focus_name = config.primary_strategy
    rows: list[dict[str, object]] = []
    cagr_gains: list[float] = []
    current_changes: list[float] = []
    for policy in policies:
        focus = metrics[metrics["strategy"].eq(focus_name) & metrics["policy"].eq(policy)].iloc[0]
        family = metrics[metrics["policy"].eq(policy)]
        focus_folds = folds[folds["strategy"].eq(focus_name) & folds["policy"].eq(policy)]
        policy_crises = crises[crises["policy"].eq(policy)]
        cost = costs[
            costs["strategy"].eq(focus_name)
            & costs["policy"].eq(policy)
            & costs["cost_bps"].eq(20.0)
        ].iloc[0]
        current_row = current[
            current["strategy"].eq(focus_name) & current["policy"].eq(policy)
        ].iloc[0]
        cagr_gain = float(focus["cagr_delta_vs_native"])
        current_change = float(current_row["defensive_weight_change"])
        cagr_gains.append(cagr_gain)
        current_changes.append(abs(current_change))
        row = {
            "policy": policy,
            "focus_strategy": focus_name,
            "focus_cagr_gain": cagr_gain,
            "family_median_cagr_gain": float(family["cagr_delta_vs_native"].median()),
            "focus_max_drawdown_delta_vs_native": float(focus["max_drawdown_delta_vs_native"]),
            "positive_focus_folds": int(focus_folds["cagr_delta_vs_native"].gt(0.0).sum()),
            "crisis_within_1pp_rate": float(
                policy_crises["max_drawdown_delta_vs_native"].ge(-0.01).mean()
            ),
            "focus_cagr_gain_at_20bps": float(cost["cagr_delta_vs_native"]),
            "current_defensive_weight_change": current_change,
        }
        row.update(
            {
                "all_four_folds_positive": row["positive_focus_folds"] == 4,
                "family_cagr_positive": row["family_median_cagr_gain"] > 0.0,
                "focus_drawdown_within_50bps": row["focus_max_drawdown_delta_vs_native"] >= -0.005,
                "crisis_90pct_within_1pp": row["crisis_within_1pp_rate"] >= 0.90,
                "positive_at_20bps": row["focus_cagr_gain_at_20bps"] > 0.0,
                "current_change_within_5pp": abs(row["current_defensive_weight_change"]) <= 0.05,
            }
        )
        rows.append(row)
    monotonic = all(later >= earlier for earlier, later in pairwise(cagr_gains)) and all(
        later >= earlier for earlier, later in pairwise(current_changes)
    )
    for row in rows:
        row["ladder_monotonic"] = monotonic
        row["implementation_safety_passed"] = all(
            bool(row[key])
            for key in (
                "all_four_folds_positive",
                "family_cagr_positive",
                "focus_drawdown_within_50bps",
                "crisis_90pct_within_1pp",
                "positive_at_20bps",
                "current_change_within_5pp",
                "ladder_monotonic",
            )
        )
        row["allocation_authority"] = 0.0
    return pd.DataFrame(rows)


def run_native_timing_progression_research(
    baseline_run: BaselineRun,
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_NATIVE_TIMING_PROGRESSION_DIR,
) -> NativeTimingProgressionRun:
    panel, strategy_families = build_warning_progression_panel(baseline_run)
    predictions, diagnostics = build_conditional_oos_predictions(panel)
    results, weights = build_progression_policy_results(
        baseline_run,
        config,
        strategy_families,
        predictions,
    )
    metrics, folds, crises, costs = evaluate_progression_policies(
        baseline_run,
        config,
        strategy_families,
        predictions,
        results,
        weights,
    )
    current = build_current_progression_read(baseline_run, config, panel)
    gates = build_progression_gates(
        config,
        diagnostics,
        metrics,
        folds,
        crises,
        costs,
    )
    rollout_ladder = build_rollout_ladder(
        config,
        metrics,
        folds,
        crises,
        costs,
        current,
    )
    frames = {
        "warning_panel": panel,
        "oos_predictions": predictions,
        "model_diagnostics": diagnostics,
        "policy_metrics": metrics,
        "fold_metrics": folds,
        "crisis_metrics": crises,
        "cost_sensitivity": costs,
        "current_read": current,
        "promotion_gates": gates,
        "rollout_ladder": rollout_ladder,
    }
    paths = write_progression_outputs(
        output_dir=output_dir,
        config=config,
        prices=baseline_run.prices,
        frames=frames,
    )
    return NativeTimingProgressionRun(
        warning_panel=panel,
        oos_predictions=predictions,
        model_diagnostics=diagnostics,
        policy_metrics=metrics,
        fold_metrics=folds,
        crisis_metrics=crises,
        cost_sensitivity=costs,
        current_read=current,
        promotion_gates=gates,
        rollout_ladder=rollout_ladder,
        output_paths=paths,
    )


def write_progression_outputs(
    *,
    output_dir: str | Path,
    config: BotConfig,
    prices: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = root / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    gates = frames["promotion_gates"]
    diagnostics = frames["model_diagnostics"]
    best = gates.iloc[0]
    summary = root / "summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Native Timing Progression Research",
                "",
                "Status: retrospective second-wave research; allocation authority is 0%.",
                "",
                f"- Conditional model mean Brier improvement: "
                f"{diagnostics['brier_improvement_vs_warning_base_rate'].mean():.4f}.",
                f"- Best policy: `{best['policy']}`.",
                f"- Gates passed: {int(best['gate_pass_count'])}/10.",
                f"- Failed gates: {best['failed_gates'] or 'none'}.",
                f"- Focus CAGR delta versus native: "
                f"{float(best['focus_cagr_delta_vs_native']):.2%}.",
                f"- Focus CAGR delta versus mild: {float(best['focus_cagr_delta_vs_mild']):.2%}.",
                f"- Focus drawdown delta versus native: "
                f"{float(best['focus_max_drawdown_delta_vs_native']):.2%}.",
                f"- Focus drawdown delta versus mild: "
                f"{float(best['focus_max_drawdown_delta_vs_mild']):.2%}.",
                "",
                "No same-history result can authorize live allocation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths["summary"] = summary
    manifest = write_research_manifest(
        root,
        study="native_timing_progression_research",
        config=config,
        prices=prices,
        oos_evidence={
            "method": "retrospective_same_history",
            "completed": True,
            "parameters_frozen_before_evaluation": False,
            "evaluation_used_for_model_selection": True,
            "primary_metric_pre_registered": True,
            "selection_cutoff": "2026-07-24",
            "evaluation_start": "2015-01-01",
            "evaluation_end": "2026-07-24",
            "observation_count": len(prices),
            "minimum_observations": 252,
        },
        parameters={
            "warning_defense_floor": WARNING_DEFENSE_FLOOR,
            "warning_defense_rise": WARNING_DEFENSE_RISE,
            "warning_model_c": WARNING_MODEL_C,
            "extra_execution_lag": EXTRA_EXECUTION_LAG,
            "mild_parameters": MILD_PARAMETERS,
            "policy_roster": list(POLICY_ROSTER),
            "progression_features": list(PROGRESSION_FEATURES),
            "automatic_promotion_allowed": False,
            "allocation_authority": 0.0,
        },
        artifacts=[path.name for path in paths.values()],
    )
    paths["manifest"] = manifest
    return paths
