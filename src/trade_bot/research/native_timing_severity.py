from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import BotConfig
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.defensive_bias_calibration import (
    _result_from_execution_weights,
    _return_slice_stats,
    comparable_strategy_family,
)
from trade_bot.research.defensive_correction_search import (
    effective_defensive_weight_path,
)
from trade_bot.research.native_timing_hazard import (
    OUTER_FOLDS,
    apply_continuous_defense_budget,
)
from trade_bot.research.native_timing_progression import (
    EXTRA_EXECUTION_LAG,
    _policy_target,
    build_conditional_oos_predictions,
    build_warning_progression_panel,
    evaluate_progression_policies,
)

DEFAULT_NATIVE_TIMING_SEVERITY_DIR = Path("reports/native_timing_severity")
POLICY_ROSTER = (
    "mild_frozen",
    "joint_severe_restore_10",
    "joint_severe_restore_15",
    "drawdown_vol_restore_10",
    "graduated_joint_restore",
    "extreme_native_veto",
)


@dataclass(frozen=True)
class NativeTimingSeverityRun:
    policy_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    crisis_metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    block_bootstrap: pd.DataFrame
    activation_summary: pd.DataFrame
    activation_episodes: pd.DataFrame
    execution_sensitivity: pd.DataFrame
    current_read: pd.DataFrame
    shadow_challenger: pd.DataFrame
    promotion_gates: pd.DataFrame
    output_paths: dict[str, Path]


def severity_policy_target(
    *,
    policy: str,
    base_defense: pd.Series,
    mild_probability: pd.Series,
    break_count: pd.Series,
    spy_drawdown: pd.Series,
    realized_volatility: pd.Series,
    high_volatility_threshold: pd.Series,
    non_deterioration_age: pd.Series,
) -> pd.Series:
    mild = _policy_target(
        policy="mild_frozen",
        base_defense=base_defense,
        probability=mild_probability,
        excess_probability=pd.Series(0.0, index=base_defense.index),
        break_count=break_count,
        non_deterioration_age=non_deterioration_age,
    )
    confirmation = break_count.ge(2.0)
    extreme_confirmation = break_count.ge(3.0)
    drawdown = spy_drawdown.le(-0.05)
    deep_drawdown = spy_drawdown.le(-0.10)
    high_volatility = realized_volatility.ge(high_volatility_threshold)
    joint = confirmation & drawdown & high_volatility
    if policy == "mild_frozen":
        target = mild
    elif policy == "joint_severe_restore_10":
        target = mild + 0.10 * joint.astype(float)
    elif policy == "joint_severe_restore_15":
        target = mild + 0.15 * joint.astype(float)
    elif policy == "drawdown_vol_restore_10":
        active = deep_drawdown | (drawdown & high_volatility)
        target = mild + 0.10 * active.astype(float)
    elif policy == "graduated_joint_restore":
        count = confirmation.astype(int) + drawdown.astype(int) + high_volatility.astype(int)
        addition = pd.Series(0.0, index=mild.index)
        addition.loc[count.eq(2)] = 0.05
        addition.loc[count.ge(3)] = 0.10
        target = mild + addition
    elif policy == "extreme_native_veto":
        active = extreme_confirmation & deep_drawdown
        native_floor = pd.concat([mild, base_defense], axis=1).max(axis=1)
        target = mild.where(~active, native_floor)
    else:
        raise ValueError(f"Unknown severity policy: {policy}")
    return target.clip(0.0, 1.0)


def build_severity_policy_results(
    baseline_run: BaselineRun,
    config: BotConfig,
    strategy_families: dict[str, str],
    predictions: pd.DataFrame,
    *,
    extra_execution_lag: int = EXTRA_EXECUTION_LAG,
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
        fold = predictions[
            predictions["outer_fold"].eq(fold_name) & predictions["family"].eq("i111")
        ].copy()
        if fold.empty:
            continue
        start = max(pd.Timestamp(start_text), pd.Timestamp(fold["origin_date"].min()))
        end = min(pd.Timestamp(end_text), pd.Timestamp(fold["origin_date"].max()))
        index = prices.index[prices.index.to_series().between(start, end).to_numpy()]
        weekly = fold.set_index("origin_date")[
            [
                "mild_base_rate_probability",
                "break_count",
                "non_deterioration_age_weeks",
                "spy_drawdown_252",
                "realized_vol_21",
                "training_realized_vol_q75",
            ]
        ].sort_index()
        state = weekly.reindex(index).ffill()
        for strategy in strategies:
            base = baseline_run.results[strategy]
            base_weights = base.weights.loc[index]
            base_defense = effective_defensive_weight_path(base_weights)
            for policy in POLICY_ROSTER:
                target = severity_policy_target(
                    policy=policy,
                    base_defense=base_defense,
                    mild_probability=state["mild_base_rate_probability"],
                    break_count=state["break_count"],
                    spy_drawdown=state["spy_drawdown_252"],
                    realized_volatility=state["realized_vol_21"],
                    high_volatility_threshold=state["training_realized_vol_q75"],
                    non_deterioration_age=state["non_deterioration_age_weeks"],
                )
                target = target.shift(extra_execution_lag).fillna(base_defense)
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


def build_severity_current_read(
    baseline_run: BaselineRun,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    latest = panel.sort_values("origin_date").groupby("family", as_index=False).tail(1)
    matured = panel[panel["forward_break"].notna()]
    mild_base_rate = float(
        (matured["forward_break"] * matured["episode_weight"]).sum()
        / matured["episode_weight"].sum()
    )
    vol_threshold = float(pd.to_numeric(matured["realized_vol_21"], errors="coerce").quantile(0.75))
    rows: list[dict[str, object]] = []
    for _, state in latest[latest["family"].eq("i111")].iterrows():
        for strategy in baseline_run.results:
            if comparable_strategy_family(strategy) != "i111":
                continue
            base_weights = baseline_run.results[strategy].weights.tail(1)
            base_defense = effective_defensive_weight_path(base_weights)
            for policy in POLICY_ROSTER:
                target = severity_policy_target(
                    policy=policy,
                    base_defense=base_defense,
                    mild_probability=pd.Series([mild_base_rate], index=base_weights.index),
                    break_count=pd.Series([float(state["break_count"])], index=base_weights.index),
                    spy_drawdown=pd.Series(
                        [float(state["spy_drawdown_252"])], index=base_weights.index
                    ),
                    realized_volatility=pd.Series(
                        [float(state["realized_vol_21"])], index=base_weights.index
                    ),
                    high_volatility_threshold=pd.Series([vol_threshold], index=base_weights.index),
                    non_deterioration_age=pd.Series(
                        [float(state["non_deterioration_age_weeks"])],
                        index=base_weights.index,
                    ),
                )
                rows.append(
                    {
                        "market_date": base_weights.index[-1],
                        "strategy": strategy,
                        "policy": policy,
                        "break_count": state["break_count"],
                        "spy_drawdown_252": state["spy_drawdown_252"],
                        "realized_vol_21": state["realized_vol_21"],
                        "realized_vol_q75": vol_threshold,
                        "base_defensive_weight": float(base_defense.iloc[0]),
                        "target_defensive_weight": float(target.iloc[0]),
                        "defensive_weight_change": float(target.iloc[0] - base_defense.iloc[0]),
                        "allocation_authority": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def build_severity_gates(
    config: BotConfig,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    crises: pd.DataFrame,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    focus_name = config.primary_strategy
    mild = metrics[metrics["strategy"].eq(focus_name) & metrics["policy"].eq("mild_frozen")].iloc[0]
    mild_gain = float(mild["cagr_delta_vs_native"])
    rows: list[dict[str, object]] = []
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


def build_severity_block_bootstrap(
    baseline_run: BaselineRun,
    config: BotConfig,
    results: dict[tuple[str, str], BacktestResult],
    predictions: pd.DataFrame,
    *,
    block_sessions: int = 63,
    simulations: int = 1_000,
    seed: int = 2_409,
) -> pd.DataFrame:
    strategy = config.primary_strategy
    start = pd.Timestamp(predictions["origin_date"].min())
    end = pd.Timestamp(predictions["origin_date"].max())
    native = baseline_run.results[strategy].returns.loc[start:end].fillna(0.0).to_numpy()
    mild = results[(strategy, "mild_frozen")].returns.loc[start:end].fillna(0.0).to_numpy()
    rows: list[dict[str, object]] = []
    generator = np.random.default_rng(seed)
    for policy in POLICY_ROSTER:
        candidate = results[(strategy, policy)].returns.loc[start:end].fillna(0.0).to_numpy()
        length = min(len(native), len(mild), len(candidate))
        paired = np.column_stack([native[:length], mild[:length], candidate[:length]])
        starts = np.arange(max(1, length - block_sessions + 1))
        blocks_needed = int(np.ceil(length / block_sessions))
        cagr_native: list[float] = []
        cagr_mild: list[float] = []
        dd_native: list[float] = []
        dd_mild: list[float] = []
        for _simulation in range(simulations):
            chosen = generator.choice(starts, size=blocks_needed, replace=True)
            sampled = np.concatenate(
                [paired[position : position + block_sessions] for position in chosen],
                axis=0,
            )[:length]
            equity = np.cumprod(1.0 + sampled, axis=0)
            cagr = equity[-1] ** (252.0 / length) - 1.0
            drawdown = np.min(
                equity / np.maximum.accumulate(equity, axis=0) - 1.0,
                axis=0,
            )
            cagr_native.append(float(cagr[2] - cagr[0]))
            cagr_mild.append(float(cagr[2] - cagr[1]))
            dd_native.append(float(drawdown[2] - drawdown[0]))
            dd_mild.append(float(drawdown[2] - drawdown[1]))
        cagr_native_array = np.asarray(cagr_native)
        cagr_mild_array = np.asarray(cagr_mild)
        dd_native_array = np.asarray(dd_native)
        dd_mild_array = np.asarray(dd_mild)
        rows.append(
            {
                "policy": policy,
                "block_sessions": block_sessions,
                "simulations": simulations,
                "cagr_delta_vs_native_p05": np.quantile(cagr_native_array, 0.05),
                "cagr_delta_vs_native_p50": np.quantile(cagr_native_array, 0.50),
                "cagr_delta_vs_native_p95": np.quantile(cagr_native_array, 0.95),
                "probability_cagr_positive_vs_native": np.mean(cagr_native_array > 0.0),
                "cagr_delta_vs_mild_p05": np.quantile(cagr_mild_array, 0.05),
                "cagr_delta_vs_mild_p50": np.quantile(cagr_mild_array, 0.50),
                "cagr_delta_vs_mild_p95": np.quantile(cagr_mild_array, 0.95),
                "max_drawdown_delta_vs_native_p05": np.quantile(dd_native_array, 0.05),
                "max_drawdown_delta_vs_native_p50": np.quantile(dd_native_array, 0.50),
                "probability_dd_damage_over_1pp_vs_native": np.mean(dd_native_array < -0.01),
                "max_drawdown_delta_vs_mild_p05": np.quantile(dd_mild_array, 0.05),
                "max_drawdown_delta_vs_mild_p50": np.quantile(dd_mild_array, 0.50),
                "probability_dd_improves_vs_mild": np.mean(dd_mild_array > 0.0),
            }
        )
    return pd.DataFrame(rows)


def build_severity_activation_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions[predictions["family"].eq("i111")].copy()
    confirmation = data["break_count"].ge(2.0)
    drawdown = data["spy_drawdown_252"].le(-0.05)
    deep_drawdown = data["spy_drawdown_252"].le(-0.10)
    high_volatility = data["realized_vol_21"].ge(data["training_realized_vol_q75"])
    conditions = {
        "joint_severe": confirmation & drawdown & high_volatility,
        "drawdown_and_high_volatility": drawdown & high_volatility,
        "deep_drawdown": deep_drawdown,
        "extreme_confirmed_deep": data["break_count"].ge(3.0) & deep_drawdown,
    }
    return pd.DataFrame(
        [
            {
                "state": name,
                "weekly_observations": int(active.sum()),
                "weekly_active_rate": float(active.mean()),
                "first_active_date": (
                    pd.to_datetime(data.loc[active, "origin_date"]).min()
                    if active.any()
                    else pd.NaT
                ),
                "last_active_date": (
                    pd.to_datetime(data.loc[active, "origin_date"]).max()
                    if active.any()
                    else pd.NaT
                ),
            }
            for name, active in conditions.items()
        ]
    )


def build_severity_activation_episodes(predictions: pd.DataFrame) -> pd.DataFrame:
    data = (
        predictions[predictions["family"].eq("i111")]
        .sort_values("origin_date")
        .drop_duplicates("origin_date")
        .copy()
    )
    active = (
        data["break_count"].ge(2.0)
        & data["spy_drawdown_252"].le(-0.05)
        & data["realized_vol_21"].ge(data["training_realized_vol_q75"])
    )
    starts = active & ~active.shift(fill_value=False)
    data["activation_episode"] = ""
    episode_number = starts.cumsum()
    data.loc[active, "activation_episode"] = "joint_severe:" + episode_number.loc[active].astype(
        str
    )
    activated = data[active].copy()
    if activated.empty:
        return pd.DataFrame()
    episodes = (
        activated.groupby("activation_episode", as_index=False)
        .agg(
            first_active_date=("origin_date", "min"),
            last_active_date=("origin_date", "max"),
            active_weeks=("origin_date", "size"),
            maximum_break_count=("break_count", "max"),
            worst_spy_drawdown=("spy_drawdown_252", "min"),
            worst_forward_drawdown=("forward_worst_drawdown", "min"),
            realized_break=("forward_break", "max"),
        )
        .sort_values("first_active_date")
    )
    episodes["share_of_active_weeks"] = episodes["active_weeks"] / episodes["active_weeks"].sum()
    return episodes


def build_severity_execution_sensitivity(
    baseline_run: BaselineRun,
    config: BotConfig,
    families: dict[str, str],
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    focus = config.primary_strategy
    start = pd.Timestamp(predictions["origin_date"].min())
    end = pd.Timestamp(predictions["origin_date"].max())
    native = baseline_run.results[focus]
    native_stats = _return_slice_stats(native.returns, start, end)
    rows: list[dict[str, object]] = []
    for lag in (0, 1, 2):
        results, _weights = build_severity_policy_results(
            baseline_run,
            config,
            families,
            predictions,
            extra_execution_lag=lag,
        )
        mild_stats = _return_slice_stats(
            results[(focus, "mild_frozen")].returns,
            start,
            end,
        )
        candidate_stats = _return_slice_stats(
            results[(focus, "joint_severe_restore_10")].returns,
            start,
            end,
        )
        rows.append(
            {
                "extra_execution_lag_sessions": lag,
                "cagr_delta_vs_native": float(candidate_stats["annualized_return"])
                - float(native_stats["annualized_return"]),
                "cagr_delta_vs_mild": float(candidate_stats["annualized_return"])
                - float(mild_stats["annualized_return"]),
                "max_drawdown_delta_vs_native": float(candidate_stats["max_drawdown"])
                - float(native_stats["max_drawdown"]),
                "max_drawdown_delta_vs_mild": float(candidate_stats["max_drawdown"])
                - float(mild_stats["max_drawdown"]),
            }
        )
    return pd.DataFrame(rows)


def build_severity_shadow_challenger(
    config: BotConfig,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    execution: pd.DataFrame,
    activation_episodes: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    policy = "joint_severe_restore_10"
    focus = metrics[
        metrics["strategy"].eq(config.primary_strategy) & metrics["policy"].eq(policy)
    ].iloc[0]
    sampled = bootstrap[bootstrap["policy"].eq(policy)].iloc[0]
    current_focus = current[
        current["strategy"].eq(config.primary_strategy) & current["policy"].eq(policy)
    ].iloc[0]
    lag_positive = bool(execution["cagr_delta_vs_native"].gt(0.0).all())
    largest_episode_share = (
        float(activation_episodes["share_of_active_weeks"].max())
        if not activation_episodes.empty
        else np.nan
    )
    activation_diversified = bool(not activation_episodes.empty and largest_episode_share < 0.75)
    diagnostic_shadow_eligible = bool(
        lag_positive and float(sampled["cagr_delta_vs_native_p05"]) > 0.0 and activation_diversified
    )
    return pd.DataFrame(
        [
            {
                "candidate": "i111_joint_severity_restore_v2_observation",
                "status": (
                    "prospective_near_miss_shadow"
                    if diagnostic_shadow_eligible
                    else "research_near_miss"
                ),
                "promotion_eligible": False,
                "policy": policy,
                "focus_strategy": config.primary_strategy,
                "observation_start_after_market_date": current_focus["market_date"],
                "retrospective_focus_cagr_delta_vs_native": focus["cagr_delta_vs_native"],
                "retrospective_focus_cagr_delta_vs_mild": focus["cagr_delta_vs_mild"],
                "retrospective_focus_drawdown_delta_vs_native": focus[
                    "max_drawdown_delta_vs_native"
                ],
                "retrospective_focus_drawdown_delta_vs_mild": focus["max_drawdown_delta_vs_mild"],
                "bootstrap_probability_cagr_positive_vs_native": sampled[
                    "probability_cagr_positive_vs_native"
                ],
                "bootstrap_probability_dd_improves_vs_mild": sampled[
                    "probability_dd_improves_vs_mild"
                ],
                "bootstrap_probability_dd_damage_over_1pp_vs_native": sampled[
                    "probability_dd_damage_over_1pp_vs_native"
                ],
                "all_execution_lags_positive_vs_native": lag_positive,
                "activation_episode_count": len(activation_episodes),
                "largest_activation_episode_share": largest_episode_share,
                "base_defensive_weight_at_freeze": current_focus["base_defensive_weight"],
                "shadow_defensive_weight_at_freeze": current_focus["target_defensive_weight"],
                "allocation_authority": 0.0,
                "automatic_promotion_allowed": False,
                "required_evidence": (
                    "new prospective joint-severity observations; no same-history retuning"
                ),
            }
        ]
    )


def freeze_severity_shadow_challenger(
    output_dir: str | Path,
    proposed: pd.DataFrame,
) -> pd.DataFrame:
    path = Path(output_dir) / "shadow_challenger.csv"
    if not path.exists():
        return proposed
    existing = pd.read_csv(path)
    if existing.empty:
        return proposed
    candidate = str(proposed.iloc[0]["candidate"])
    frozen = existing[existing["candidate"].astype(str).eq(candidate)]
    if frozen.empty:
        return proposed
    # Preserve values captured when the observation was frozen. Diagnostic
    # schema may grow, but later runs must not silently rewrite the original
    # retrospective metrics, status, or freeze-date allocation.
    frozen_row = frozen.head(1).copy()
    proposed_row = proposed.head(1)
    for column in proposed_row.columns.difference(frozen_row.columns):
        frozen_row[column] = proposed_row.iloc[0][column]
    return frozen_row


def run_native_timing_severity_research(
    baseline_run: BaselineRun,
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_NATIVE_TIMING_SEVERITY_DIR,
) -> NativeTimingSeverityRun:
    panel, families = build_warning_progression_panel(baseline_run)
    predictions, _diagnostics = build_conditional_oos_predictions(panel)
    results, weights = build_severity_policy_results(baseline_run, config, families, predictions)
    metrics, folds, crises, costs = evaluate_progression_policies(
        baseline_run,
        config,
        families,
        predictions,
        results,
        weights,
        policy_roster=POLICY_ROSTER,
    )
    bootstrap = build_severity_block_bootstrap(
        baseline_run,
        config,
        results,
        predictions,
    )
    activation = build_severity_activation_summary(predictions)
    activation_episodes = build_severity_activation_episodes(predictions)
    execution = build_severity_execution_sensitivity(
        baseline_run,
        config,
        families,
        predictions,
    )
    current = build_severity_current_read(baseline_run, panel)
    shadow = freeze_severity_shadow_challenger(
        output_dir,
        build_severity_shadow_challenger(
            config,
            metrics,
            bootstrap,
            execution,
            activation_episodes,
            current,
        ),
    )
    gates = build_severity_gates(config, metrics, folds, crises, costs)
    frames = {
        "policy_metrics": metrics,
        "fold_metrics": folds,
        "crisis_metrics": crises,
        "cost_sensitivity": costs,
        "block_bootstrap": bootstrap,
        "activation_summary": activation,
        "activation_episodes": activation_episodes,
        "execution_sensitivity": execution,
        "current_read": current,
        "shadow_challenger": shadow,
        "promotion_gates": gates,
    }
    paths = write_severity_outputs(
        output_dir=output_dir,
        config=config,
        prices=baseline_run.prices,
        frames=frames,
    )
    return NativeTimingSeverityRun(
        policy_metrics=metrics,
        fold_metrics=folds,
        crisis_metrics=crises,
        cost_sensitivity=costs,
        block_bootstrap=bootstrap,
        activation_summary=activation,
        activation_episodes=activation_episodes,
        execution_sensitivity=execution,
        current_read=current,
        shadow_challenger=shadow,
        promotion_gates=gates,
        output_paths=paths,
    )


def write_severity_outputs(
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
    best = frames["promotion_gates"].iloc[0]
    summary = root / "summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Native Timing Joint-Severity Research",
                "",
                "Status: post-hoc third-wave research; allocation authority is 0%.",
                "",
                f"- Best policy: `{best['policy']}`.",
                f"- Gates passed: {int(best['gate_pass_count'])}/9.",
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
        study="native_timing_joint_severity_research",
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
            "policy_roster": list(POLICY_ROSTER),
            "extra_execution_lag": EXTRA_EXECUTION_LAG,
            "automatic_promotion_allowed": False,
            "allocation_authority": 0.0,
        },
        artifacts=[path.name for path in paths.values()],
    )
    paths["manifest"] = manifest
    return paths
