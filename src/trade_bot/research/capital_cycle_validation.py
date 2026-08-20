from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trade_bot.research.capital_cycle_break import (
    CAPITAL_CYCLE_FAMILIES,
    run_capital_cycle_break_monitor,
)

DEFAULT_CAPITAL_CYCLE_VALIDATION_DIR = Path("reports/capital_cycle_validation")
MARKET_HORIZONS: Final[dict[str, tuple[int, float]]] = {
    "theme_break_10_63d": (63, -0.10),
    "theme_break_20_126d": (126, -0.20),
}
ECONOMIC_HORIZONS: Final[dict[str, int]] = {
    "economic_rollover_2q": 126,
    "economic_rollover_4q": 252,
}
MODEL_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "frozen_trade_bot": ("frozen_trade_bot_risk_reduction",),
    # Frozen comparator only: these features are owned by the existing Trade-bot
    # stack and are not candidates for a new capital-cycle signal.
    "existing_market": (
        "family__theme_market_break",
        "family__financing_reflexivity",
        "family__margin_leverage",
        "family__rates_capital_competition",
        "family__equity_concentration",
    ),
    "upstream_vulnerability": (
        "direct__capital_intensity_overbuild",
        "direct__financing_reflexivity",
        "direct__monetization_efficiency",
    ),
    "upstream_rollover": (
        "direct__infrastructure_demand_rollover",
        "direct__financing_reflexivity",
        "direct__monetization_efficiency",
    ),
    "upstream_combined": (
        "direct__capital_intensity_overbuild",
        "direct__financing_reflexivity",
        "direct__monetization_efficiency",
        "direct__infrastructure_demand_rollover",
    ),
    "sequential_state": (
        "state_ordinal",
        "real_economy_confirmations",
        "independent_rollover_confirmations",
    ),
    "full_incremental": (
        "family__theme_market_break",
        "family__financing_reflexivity",
        "family__margin_leverage",
        "family__rates_capital_competition",
        "family__equity_concentration",
        "direct__capital_intensity_overbuild",
        "direct__financing_reflexivity",
        "direct__monetization_efficiency",
        "direct__infrastructure_demand_rollover",
        "state_ordinal",
        "real_economy_confirmations",
        "independent_rollover_confirmations",
    ),
}
REGULARIZATION_GRID: Final[tuple[float, ...]] = (0.05, 0.20, 1.0)
STATE_ORDINAL: Final[dict[str, float]] = {
    "insufficient_data": np.nan,
    "normal": 0.0,
    "ordinary_correction_or_early_rollover": 0.25,
    "buildout_fragile": 0.35,
    "prebreak_pressure": 0.60,
    "rollover_confirming": 0.80,
    "break_confirmed": 1.0,
}


@dataclass(frozen=True)
class CapitalCycleValidationRun:
    origin_panel: pd.DataFrame
    oos_predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    episode_diagnostics: pd.DataFrame
    leave_episode_out: pd.DataFrame
    uncertainty: pd.DataFrame
    ablation_summary: pd.DataFrame
    utility_summary: pd.DataFrame
    stage_gates: pd.DataFrame
    readiness: pd.DataFrame
    output_paths: dict[str, Path]


def run_capital_cycle_validation(
    *,
    prices: pd.DataFrame,
    macro_data: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    economic_events: pd.DataFrame | None = None,
    episode_manifest: pd.DataFrame | None = None,
    frozen_trade_bot_baseline: pd.DataFrame | None = None,
    theme_ticker: str = "QQQ",
    origin_step_days: int = 21,
    min_history_days: int = 756,
    output_dir: str | Path = DEFAULT_CAPITAL_CYCLE_VALIDATION_DIR,
) -> CapitalCycleValidationRun:
    """Run a prior-only, episode-aware validation of the capital-cycle monitor."""

    origin_panel = build_capital_cycle_origin_panel(
        prices=prices,
        macro_data=macro_data,
        fundamentals=fundamentals,
        events=events,
        economic_events=economic_events,
        episode_manifest=episode_manifest,
        frozen_trade_bot_baseline=frozen_trade_bot_baseline,
        theme_ticker=theme_ticker,
        origin_step_days=origin_step_days,
        min_history_days=min_history_days,
    )
    economic_predictions, economic_diagnostics = build_nested_capital_cycle_predictions(
        origin_panel,
        target_names=tuple(ECONOMIC_HORIZONS),
    )
    economic_leave_out = build_leave_episode_out_diagnostics(
        origin_panel,
        target_names=tuple(ECONOMIC_HORIZONS),
    )
    stage_gates = build_economic_forecasting_gate(
        origin_panel,
        economic_diagnostics,
        economic_leave_out,
    )
    economic_gate_passed = bool(stage_gates.iloc[0]["economic_forecasting_gate_passed"])
    if economic_gate_passed:
        market_predictions, market_diagnostics = build_nested_capital_cycle_predictions(
            origin_panel,
            target_names=tuple(MARKET_HORIZONS),
        )
        market_leave_out = build_leave_episode_out_diagnostics(
            origin_panel,
            target_names=tuple(MARKET_HORIZONS),
        )
    else:
        market_predictions = pd.DataFrame()
        market_diagnostics = pd.DataFrame()
        market_leave_out = pd.DataFrame()
    predictions = _concat_nonempty(economic_predictions, market_predictions)
    diagnostics = _concat_nonempty(economic_diagnostics, market_diagnostics)
    leave_episode_out = _concat_nonempty(economic_leave_out, market_leave_out)
    episode_diagnostics = build_episode_diagnostics(predictions)
    uncertainty = build_episode_bootstrap_uncertainty(predictions)
    ablation_summary = build_ablation_summary(diagnostics)
    utility_summary = build_probability_utility_summary(predictions)
    readiness = build_validation_readiness(
        origin_panel,
        diagnostics,
        episode_diagnostics,
        stage_gates=stage_gates,
    )
    frames = {
        "origin_panel": origin_panel,
        "oos_predictions": predictions,
        "diagnostics": diagnostics,
        "episode_diagnostics": episode_diagnostics,
        "leave_episode_out": leave_episode_out,
        "uncertainty": uncertainty,
        "ablation_summary": ablation_summary,
        "utility_summary": utility_summary,
        "stage_gates": stage_gates,
        "readiness": readiness,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output / f"capital_cycle_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return CapitalCycleValidationRun(output_paths=paths, **frames)


def build_capital_cycle_origin_panel(
    *,
    prices: pd.DataFrame,
    macro_data: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    economic_events: pd.DataFrame | None = None,
    episode_manifest: pd.DataFrame | None = None,
    frozen_trade_bot_baseline: pd.DataFrame | None = None,
    theme_ticker: str = "QQQ",
    origin_step_days: int = 21,
    min_history_days: int = 756,
) -> pd.DataFrame:
    clean_prices = _clean_prices(prices)
    if theme_ticker not in clean_prices:
        raise ValueError(f"Theme ticker {theme_ticker!r} is not present in prices.")
    if len(clean_prices) <= min_history_days:
        return pd.DataFrame()
    step = max(1, int(origin_step_days))
    maximum_horizon = max(
        [horizon for horizon, _threshold in MARKET_HORIZONS.values()]
        + list(ECONOMIC_HORIZONS.values())
    )
    rows: list[dict[str, object]] = []
    origin_specs = _build_origin_specs(
        clean_prices,
        episode_manifest,
        step=step,
        min_history_days=min_history_days,
        default_ticker=theme_ticker,
    )
    for (
        location,
        episode_name,
        episode_cluster,
        episode_ticker,
        episode_definition,
    ) in origin_specs:
        origin_date = pd.Timestamp(clean_prices.index[location])
        effective_ticker = episode_ticker if episode_ticker in clean_prices else theme_ticker
        monitor = run_capital_cycle_break_monitor(
            prices=clean_prices.iloc[: location + 1],
            macro_data=_truncate_indexed_frame(macro_data, origin_date),
            fundamentals=_episode_frame(fundamentals, episode_name),
            events=_episode_frame(events, episode_name),
            as_of=origin_date,
        )
        summary = monitor.summary.iloc[0]
        family_index = monitor.families.set_index("family")
        row: dict[str, object] = {
            "origin_date": origin_date,
            "episode": episode_name,
            "episode_cluster": episode_cluster,
            "episode_definition": episode_definition,
            "theme_ticker": effective_ticker,
            "registered_theme_ticker": episode_ticker,
            "theme_ticker_fallback_used": effective_ticker != episode_ticker,
            "capital_vulnerability": summary.get("capital_vulnerability"),
            "rollover_impulse": summary.get("rollover_impulse"),
            "vulnerability_coverage": summary.get("vulnerability_coverage"),
            "rollover_coverage": summary.get("rollover_coverage"),
            "capital_cycle_state": summary.get("capital_cycle_state"),
            "state_ordinal": STATE_ORDINAL.get(str(summary.get("capital_cycle_state")), np.nan),
            "real_economy_confirmations": summary.get("real_economy_confirmations", 0),
            "independent_rollover_confirmations": summary.get(
                "independent_rollover_confirmations", 0
            ),
            "allocation_authority": 0.0,
        }
        row.update(_frozen_baseline_at_origin(frozen_trade_bot_baseline, origin_date))
        row["vulnerability_rollover_interaction"] = _numeric_product(
            row["capital_vulnerability"], row["rollover_impulse"]
        )
        for family in CAPITAL_CYCLE_FAMILIES:
            if family not in family_index.index:
                row[f"family__{family}"] = np.nan
                row[f"direct__{family}"] = np.nan
                continue
            family_row = family_index.loc[family]
            score = pd.to_numeric(pd.Series([family_row.get("score")]), errors="coerce").iloc[0]
            row[f"family__{family}"] = score
            row[f"direct__{family}"] = (
                score if str(family_row.get("data_status")) == "direct" else np.nan
            )
        _attach_market_targets(
            row,
            prices=clean_prices,
            origin_location=location,
            theme_ticker=effective_ticker,
        )
        _attach_economic_targets(
            row,
            origin_date=origin_date,
            episode=episode_name,
            economic_events=economic_events,
            maximum_horizon=maximum_horizon,
        )
        rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel["episode_weight"] = _episode_balanced_weights(panel["episode_cluster"])
    return panel.sort_values("origin_date").reset_index(drop=True)


def build_nested_capital_cycle_predictions(
    origin_panel: pd.DataFrame,
    *,
    outer_folds: int = 4,
    target_names: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if origin_panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    prediction_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    targets = target_names or (*MARKET_HORIZONS, *ECONOMIC_HORIZONS)
    for target in targets:
        maturity_column = f"{target}_maturity_date"
        if target not in origin_panel or maturity_column not in origin_panel:
            continue
        matured = origin_panel[origin_panel[target].notna()].copy()
        if matured.empty or matured[target].nunique() < 2:
            diagnostic_rows.extend(_not_evaluable_rows(target, "target_has_one_class_or_no_rows"))
            continue
        folds = _expanding_outer_folds(matured, maturity_column, outer_folds=outer_folds)
        for fold_name, train, test in folds:
            if train.empty or test.empty or train[target].nunique() < 2:
                continue
            base_rate = _weighted_average(train[target], train["episode_weight"])
            base_probability = np.repeat(base_rate, len(test))
            base_brier = _weighted_brier(test[target], base_probability, test["episode_weight"])
            for model_name, requested_features in MODEL_FEATURES.items():
                features = _eligible_features(train, test, requested_features)
                if not _model_is_evaluable(model_name, features):
                    diagnostic_rows.append(
                        _not_evaluable_row(
                            target,
                            model_name,
                            fold_name,
                            "required_features_unavailable",
                            train,
                            test,
                            base_brier,
                        )
                    )
                    continue
                inner_train, validation = _inner_split(train, maturity_column)
                best_c = _select_regularization(
                    inner_train,
                    validation,
                    target=target,
                    features=features,
                )
                probability = _fit_probability(
                    train,
                    test,
                    target=target,
                    features=features,
                    c_value=best_c,
                )
                identity_columns = ["origin_date", "episode"]
                if "episode_cluster" in test:
                    identity_columns.append("episode_cluster")
                identity_columns.extend(["episode_weight", target, maturity_column])
                prediction = test[
                    identity_columns
                    + [
                        column
                        for column in (
                            f"{target}_forward_return",
                            f"{target}_forward_drawdown",
                        )
                        if column in test
                    ]
                ].copy()
                prediction["target"] = target
                prediction["model"] = model_name
                prediction["outer_fold"] = fold_name
                prediction["probability"] = probability
                prediction["base_rate_probability"] = base_probability
                prediction["selected_c"] = best_c
                prediction["allocation_authority"] = 0.0
                prediction_rows.append(prediction)
                diagnostic_rows.append(
                    _diagnostic_row(
                        target=target,
                        model_name=model_name,
                        fold_name=fold_name,
                        train=train,
                        test=test,
                        probability=probability,
                        base_probability=base_probability,
                        selected_c=best_c,
                        features=features,
                    )
                )
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return predictions, pd.DataFrame(diagnostic_rows)


def build_episode_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for (target, model, episode_cluster), group in predictions.groupby(
        ["target", "model", "episode_cluster"], dropna=False
    ):
        label = pd.to_numeric(group[target], errors="coerce")
        probability = pd.to_numeric(group["probability"], errors="coerce")
        base = pd.to_numeric(group["base_rate_probability"], errors="coerce")
        weights = pd.to_numeric(group["episode_weight"], errors="coerce").fillna(1.0)
        rows.append(
            {
                "target": target,
                "model": model,
                "episode": episode_cluster,
                "episode_cluster": episode_cluster,
                "origins": len(group),
                "events": int(label.sum()),
                "brier_score": _weighted_brier(label, probability, weights),
                "base_rate_brier_score": _weighted_brier(label, base, weights),
                "brier_improvement_vs_base_rate": _weighted_brier(label, base, weights)
                - _weighted_brier(label, probability, weights),
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_leave_episode_out_diagnostics(
    origin_panel: pd.DataFrame,
    *,
    target_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Test portability by refitting without each complete episode cluster."""

    if origin_panel.empty:
        return pd.DataFrame()
    rows = []
    targets = target_names or (*MARKET_HORIZONS, *ECONOMIC_HORIZONS)
    for target in targets:
        if target not in origin_panel:
            continue
        matured = origin_panel[origin_panel[target].notna()].copy()
        if matured.empty or matured[target].nunique() < 2:
            continue
        cluster_column = "episode_cluster" if "episode_cluster" in matured else "episode"
        for held_episode in matured[cluster_column].dropna().unique():
            train = matured[~matured[cluster_column].eq(held_episode)].copy()
            test = matured[matured[cluster_column].eq(held_episode)].copy()
            if train.empty or test.empty or train[target].nunique() < 2:
                continue
            base_rate = _weighted_average(train[target], train["episode_weight"])
            base_probability = np.repeat(base_rate, len(test))
            for model_name, requested_features in MODEL_FEATURES.items():
                features = _eligible_features(train, test, requested_features)
                if not _model_is_evaluable(model_name, features):
                    continue
                probability = _fit_probability(
                    train,
                    test,
                    target=target,
                    features=features,
                    c_value=0.20,
                )
                diagnostic = _diagnostic_row(
                    target=target,
                    model_name=model_name,
                    fold_name=f"leave_{held_episode}_out",
                    train=train,
                    test=test,
                    probability=probability,
                    base_probability=base_probability,
                    selected_c=0.20,
                    features=features,
                )
                diagnostic["held_out_episode"] = held_episode
                diagnostic["validation_design"] = (
                    "leave_episode_out_robustness_not_chronological_holdout"
                )
                rows.append(diagnostic)
    return pd.DataFrame(rows)


def build_episode_bootstrap_uncertainty(
    predictions: pd.DataFrame,
    *,
    draws: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap complete episode clusters, never individual origin rows."""

    if predictions.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_seed)
    rows = []
    for (target, model), group in predictions.groupby(["target", "model"]):
        cluster_column = "episode_cluster" if "episode_cluster" in group else "episode"
        episodes = list(group[cluster_column].dropna().unique())
        if len(episodes) < 2:
            continue
        samples = []
        for _ in range(max(1, int(draws))):
            selected = rng.choice(episodes, size=len(episodes), replace=True)
            pieces = [group[group[cluster_column].eq(episode)] for episode in selected]
            sample = pd.concat(pieces, ignore_index=True)
            labels = pd.to_numeric(sample[target], errors="coerce")
            weights = pd.to_numeric(sample["episode_weight"], errors="coerce").fillna(1.0)
            model_brier = _weighted_brier(labels, sample["probability"], weights)
            base_brier = _weighted_brier(labels, sample["base_rate_probability"], weights)
            samples.append(base_brier - model_brier)
        values = np.asarray(samples, dtype=float)
        rows.append(
            {
                "target": target,
                "model": model,
                "episode_clusters": len(episodes),
                "bootstrap_draws": len(values),
                "median_brier_improvement": float(np.median(values)),
                "lower_05_brier_improvement": float(np.quantile(values, 0.05)),
                "upper_95_brier_improvement": float(np.quantile(values, 0.95)),
                "positive_improvement_probability": float((values > 0).mean()),
                "resampling_unit": "episode_cluster",
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_ablation_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    evaluated = diagnostics[diagnostics.get("status", "").eq("evaluated")].copy()
    if evaluated.empty:
        return pd.DataFrame()
    rows = []
    for (target, model), group in evaluated.groupby(["target", "model"]):
        improvement = pd.to_numeric(
            group["brier_improvement_vs_base_rate"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "target": target,
                "model": model,
                "outer_folds": group["outer_fold"].nunique(),
                "median_brier_improvement_vs_base_rate": improvement.median(),
                "mean_brier_improvement_vs_base_rate": improvement.mean(),
                "positive_fold_share": float((improvement > 0).mean()),
                "median_auc": pd.to_numeric(group["auc"], errors="coerce").median(),
                "evaluation_status": "retrospective_nested_oos",
                "allocation_authority": 0.0,
            }
        )
    output = pd.DataFrame(rows)
    market_baseline = output[output["model"].eq("existing_market")][
        ["target", "median_brier_improvement_vs_base_rate"]
    ].rename(columns={"median_brier_improvement_vs_base_rate": "existing_market_improvement"})
    output = output.merge(market_baseline, on="target", how="left")
    output["incremental_improvement_vs_existing_market"] = (
        output["median_brier_improvement_vs_base_rate"]
        - output["existing_market_improvement"]
    )
    return output.sort_values(
        ["target", "median_brier_improvement_vs_base_rate"], ascending=[True, False]
    ).reset_index(drop=True)


def build_probability_utility_summary(
    predictions: pd.DataFrame,
    *,
    maximum_defensive_adjustment: float = 0.15,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    market = predictions[predictions["target"].isin(MARKET_HORIZONS)].copy()
    for (target, model), group in market.groupby(["target", "model"]):
        return_column = f"{target}_forward_return"
        drawdown_column = f"{target}_forward_drawdown"
        if return_column not in group:
            continue
        probability = pd.to_numeric(group["probability"], errors="coerce").clip(0, 1)
        defense = maximum_defensive_adjustment * probability
        forward_return = pd.to_numeric(group[return_column], errors="coerce")
        forward_drawdown = pd.to_numeric(group.get(drawdown_column), errors="coerce")
        incremental_return = -defense * forward_return
        rows.append(
            {
                "target": target,
                "model": model,
                "origins": len(group),
                "maximum_defensive_adjustment": maximum_defensive_adjustment,
                "mean_defensive_adjustment": defense.mean(),
                "mean_incremental_forward_return": incremental_return.mean(),
                "mean_avoided_drawdown_proxy": (
                    defense * (-forward_drawdown.clip(upper=0.0))
                ).mean(),
                "mean_foregone_upside": (
                    defense * forward_return.clip(lower=0.0)
                ).mean(),
                "interpretation": "overlapping-origin diagnostic; not a portfolio backtest",
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_economic_forecasting_gate(
    origin_panel: pd.DataFrame,
    diagnostics: pd.DataFrame,
    leave_episode_out: pd.DataFrame,
) -> pd.DataFrame:
    """Gate all equity timing behind portable direct economic forecasting value."""

    blockers: list[str] = []
    economic_rows = origin_panel[
        [target for target in ECONOMIC_HORIZONS if target in origin_panel]
    ] if not origin_panel.empty else pd.DataFrame()
    labeled_episodes = 0
    if not economic_rows.empty:
        label_mask = economic_rows.notna().any(axis=1)
        cluster_column = (
            "episode_cluster" if "episode_cluster" in origin_panel else "episode"
        )
        labeled_episodes = int(origin_panel.loc[label_mask, cluster_column].nunique())
    if labeled_episodes < 3:
        blockers.append("fewer_than_three_labeled_economic_episodes")
    evaluated = diagnostics[
        diagnostics.get("status", pd.Series(dtype=str)).eq("evaluated")
        & diagnostics.get("target", pd.Series(dtype=str)).isin(ECONOMIC_HORIZONS)
    ].copy()
    direct_models = {
        "upstream_vulnerability",
        "upstream_rollover",
        "upstream_combined",
        "sequential_state",
        "full_incremental",
    }
    direct = evaluated[evaluated["model"].isin(direct_models)]
    if direct.empty:
        blockers.append("no_evaluable_direct_economic_model")
    grouped = (
        direct.groupby(["target", "model"])
        .agg(
            median_improvement=("brier_improvement_vs_base_rate", "median"),
            positive_fold_share=(
                "brier_improvement_vs_base_rate",
                lambda values: float((values > 0).mean()),
            ),
        )
        .reset_index()
        if not direct.empty
        else pd.DataFrame()
    )
    candidates = (
        grouped[
            grouped["median_improvement"].gt(0)
            & grouped["positive_fold_share"].ge(0.50)
        ]
        if not grouped.empty
        else pd.DataFrame()
    )
    if candidates.empty:
        blockers.append("no_direct_model_beats_event_rate_across_folds")
    leave_direct = leave_episode_out[
        leave_episode_out.get("model", pd.Series(dtype=str)).isin(direct_models)
        & leave_episode_out.get("target", pd.Series(dtype=str)).isin(ECONOMIC_HORIZONS)
    ].copy()
    leave_positive_share = (
        float(leave_direct["brier_improvement_vs_base_rate"].gt(0).mean())
        if not leave_direct.empty
        else np.nan
    )
    if leave_direct.empty or leave_positive_share < 0.50:
        blockers.append("leave_episode_out_portability_below_half")
    return pd.DataFrame(
        [
            {
                "labeled_economic_episodes": labeled_episodes,
                "evaluated_direct_economic_fold_rows": len(direct),
                "qualifying_direct_economic_models": len(candidates),
                "leave_episode_out_positive_share": leave_positive_share,
                "economic_forecasting_gate_passed": not blockers,
                "equity_timing_unlocked": not blockers,
                "strategy_utility_unlocked": False,
                "blockers": ";".join(blockers),
                "allocation_authority": 0.0,
            }
        ]
    )


def build_validation_readiness(
    origin_panel: pd.DataFrame,
    diagnostics: pd.DataFrame,
    episode_diagnostics: pd.DataFrame,
    *,
    stage_gates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    direct_columns = [
        f"direct__{family}"
        for family in (
            "capital_intensity_overbuild",
            "monetization_efficiency",
            "infrastructure_demand_rollover",
        )
        if f"direct__{family}" in origin_panel
    ]
    direct_origins = (
        int(origin_panel[direct_columns].notna().any(axis=1).sum())
        if direct_columns and not origin_panel.empty
        else 0
    )
    evaluated = diagnostics[diagnostics.get("status", "").eq("evaluated")]
    cluster_column = "episode_cluster" if "episode_cluster" in origin_panel else "episode"
    episodes = int(origin_panel[cluster_column].nunique()) if not origin_panel.empty else 0
    concentration = np.nan
    if not episode_diagnostics.empty:
        positive = episode_diagnostics[
            episode_diagnostics["brier_improvement_vs_base_rate"] > 0
        ]
        if not positive.empty:
            diagnostic_cluster = (
                "episode_cluster" if "episode_cluster" in positive else "episode"
            )
            by_episode = positive.groupby(diagnostic_cluster)[
                "brier_improvement_vs_base_rate"
            ].sum()
            concentration = float(by_episode.max() / by_episode.sum()) if by_episode.sum() else np.nan
    blockers = []
    if direct_origins == 0:
        blockers.append("no_historical_direct_upstream_origins")
    economic_labels = [
        target
        for target in ECONOMIC_HORIZONS
        if target in origin_panel and origin_panel[target].notna().any()
    ]
    if not economic_labels:
        blockers.append("no_objective_economic_rollover_labels")
    if (
        "episode_definition" in origin_panel
        and origin_panel["episode_definition"].eq("calendar_blocks_placeholder").any()
    ):
        blockers.append("named_analogue_episode_manifest_missing")
    if episodes < 3:
        blockers.append("fewer_than_three_episode_clusters")
    if evaluated.empty:
        blockers.append("no_evaluable_nested_oos_models")
    if (
        isinstance(stage_gates, pd.DataFrame)
        and not stage_gates.empty
        and not bool(stage_gates.iloc[0].get("economic_forecasting_gate_passed", False))
    ):
        blockers.append("economic_forecasting_gate_not_passed")
    if pd.notna(concentration) and concentration > 0.50:
        blockers.append("positive_lift_concentrated_in_one_episode")
    blockers.append("prospective_shadow_not_mature")
    return pd.DataFrame(
        [
            {
                "origin_rows": len(origin_panel),
                "episode_clusters": episodes,
                "origins_with_direct_upstream_evidence": direct_origins,
                "evaluated_model_fold_rows": len(evaluated),
                "positive_evidence_episode_concentration": concentration,
                "historical_research_ready": not any(
                    blocker
                    for blocker in blockers
                    if blocker != "prospective_shadow_not_mature"
                ),
                "promotion_ready": False,
                "blockers": ";".join(blockers),
                "validation_design": "nested_expanding_walk_forward_episode_balanced",
                "allocation_authority": 0.0,
            }
        ]
    )


def _attach_market_targets(
    row: dict[str, object],
    *,
    prices: pd.DataFrame,
    origin_location: int,
    theme_ticker: str,
) -> None:
    origin_price = float(prices[theme_ticker].iloc[origin_location])
    for target, (horizon, threshold) in MARKET_HORIZONS.items():
        end = origin_location + horizon
        maturity_column = f"{target}_maturity_date"
        if end >= len(prices) or not np.isfinite(origin_price) or origin_price <= 0:
            row[target] = np.nan
            row[maturity_column] = pd.NaT
            row[f"{target}_forward_return"] = np.nan
            row[f"{target}_forward_drawdown"] = np.nan
            continue
        path = pd.to_numeric(
            prices[theme_ticker].iloc[origin_location + 1 : end + 1], errors="coerce"
        ).dropna()
        if path.empty:
            row[target] = np.nan
            row[maturity_column] = pd.NaT
            continue
        forward_return = float(path.iloc[-1] / origin_price - 1.0)
        forward_drawdown = float((path / origin_price - 1.0).min())
        row[target] = float(forward_drawdown <= threshold)
        row[maturity_column] = pd.Timestamp(prices.index[end])
        row[f"{target}_forward_return"] = forward_return
        row[f"{target}_forward_drawdown"] = forward_drawdown


def _attach_economic_targets(
    row: dict[str, object],
    *,
    origin_date: pd.Timestamp,
    episode: str,
    economic_events: pd.DataFrame | None,
    maximum_horizon: int,
) -> None:
    if not isinstance(economic_events, pd.DataFrame) or economic_events.empty:
        for target in ECONOMIC_HORIZONS:
            row[target] = np.nan
            row[f"{target}_maturity_date"] = pd.NaT
        return
    events = economic_events.copy()
    if "episode" in events:
        events = events[events["episode"].astype(str).eq(episode)]
    if "objective_label_resolved" in events and not events.empty:
        resolved = events["objective_label_resolved"].fillna(False).astype(bool)
        if not bool(resolved.any()):
            for target in ECONOMIC_HORIZONS:
                row[target] = np.nan
                row[f"{target}_maturity_date"] = pd.NaT
            return
    events["event_date"] = pd.to_datetime(events.get("event_date"), errors="coerce")
    events = events.dropna(subset=["event_date"])
    event_targets = (
        events["target"].astype(str)
        if "target" in events
        else pd.Series("economic_rollover", index=events.index)
    )
    for target, horizon in ECONOMIC_HORIZONS.items():
        maturity = origin_date + pd.offsets.BDay(horizon)
        eligible = events[
            events["event_date"].gt(origin_date)
            & events["event_date"].le(maturity)
            & event_targets.isin({target, "economic_rollover"})
        ]
        row[target] = float(not eligible.empty)
        row[f"{target}_maturity_date"] = maturity


def _expanding_outer_folds(
    frame: pd.DataFrame,
    maturity_column: str,
    *,
    outer_folds: int,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    ordered = frame.sort_values("origin_date")
    unique_dates = pd.Index(pd.to_datetime(ordered["origin_date"]).dropna().unique()).sort_values()
    if len(unique_dates) < 8:
        return []
    boundaries = np.linspace(int(len(unique_dates) * 0.50), len(unique_dates), outer_folds + 1)
    boundaries = np.unique(boundaries.astype(int))
    folds = []
    for fold_number, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True), start=1
    ):
        if start >= len(unique_dates) or end <= start:
            continue
        test_start = pd.Timestamp(unique_dates[start])
        test_end = pd.Timestamp(unique_dates[min(end - 1, len(unique_dates) - 1)])
        train = ordered[pd.to_datetime(ordered[maturity_column]).lt(test_start)].copy()
        test = ordered[pd.to_datetime(ordered["origin_date"]).between(test_start, test_end)].copy()
        folds.append((f"outer_{fold_number}", train, test))
    return folds


def _inner_split(
    train: pd.DataFrame,
    maturity_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(train["origin_date"])
    split = dates.quantile(0.75)
    inner = train[pd.to_datetime(train[maturity_column]).lt(split)].copy()
    validation = train[dates.ge(split)].copy()
    return inner, validation


def _select_regularization(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    target: str,
    features: list[str],
) -> float:
    if train.empty or validation.empty or train[target].nunique() < 2:
        return 0.20
    scored = []
    for c_value in REGULARIZATION_GRID:
        probability = _fit_probability(
            train,
            validation,
            target=target,
            features=features,
            c_value=c_value,
        )
        scored.append(
            (
                _weighted_brier(
                    validation[target], probability, validation["episode_weight"]
                ),
                c_value,
            )
        )
    return min(scored)[1]


def _fit_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    c_value: float,
) -> np.ndarray:
    if train[target].nunique() < 2:
        return np.repeat(float(train[target].mean()), len(test))
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight=None,
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(
        train[features],
        pd.to_numeric(train[target], errors="coerce").astype(int),
        model__sample_weight=pd.to_numeric(
            train["episode_weight"], errors="coerce"
        ).fillna(1.0),
    )
    return pipeline.predict_proba(test[features])[:, 1]


def _eligible_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    requested: tuple[str, ...],
) -> list[str]:
    return [
        feature
        for feature in requested
        if feature in train
        and feature in test
        and train[feature].notna().any()
        and test[feature].notna().any()
        and pd.to_numeric(train[feature], errors="coerce").nunique(dropna=True) > 1
    ]


def _model_is_evaluable(model_name: str, features: list[str]) -> bool:
    available = set(features)
    upstream = {
        "direct__capital_intensity_overbuild",
        "direct__monetization_efficiency",
        "direct__infrastructure_demand_rollover",
    }
    if model_name == "existing_market":
        return len(available) >= 2
    if model_name == "frozen_trade_bot":
        return "frozen_trade_bot_risk_reduction" in available
    if model_name in {
        "upstream_vulnerability",
        "upstream_rollover",
        "upstream_combined",
    }:
        return bool(available.intersection(upstream))
    if model_name == "sequential_state":
        return "state_ordinal" in available
    if model_name == "full_incremental":
        return bool(available.intersection(upstream)) and any(
            feature.startswith("family__") for feature in available
        )
    return bool(features)


def _diagnostic_row(
    *,
    target: str,
    model_name: str,
    fold_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    probability: np.ndarray,
    base_probability: np.ndarray,
    selected_c: float,
    features: list[str],
) -> dict[str, object]:
    labels = pd.to_numeric(test[target], errors="coerce")
    weights = pd.to_numeric(test["episode_weight"], errors="coerce").fillna(1.0)
    brier = _weighted_brier(labels, probability, weights)
    base_brier = _weighted_brier(labels, base_probability, weights)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    auc = roc_auc_score(labels, probability, sample_weight=weights) if labels.nunique() > 1 else np.nan
    return {
        "target": target,
        "model": model_name,
        "outer_fold": fold_name,
        "status": "evaluated",
        "train_origins": len(train),
        "test_origins": len(test),
        "test_episodes": test[
            "episode_cluster" if "episode_cluster" in test else "episode"
        ].nunique(),
        "event_rate": _weighted_average(labels, weights),
        "brier_score": brier,
        "base_rate_brier_score": base_brier,
        "brier_improvement_vs_base_rate": base_brier - brier,
        "log_loss": log_loss(labels, clipped, sample_weight=weights, labels=[0, 1]),
        "auc": auc,
        "selected_c": selected_c,
        "features": ";".join(features),
        "allocation_authority": 0.0,
    }


def _not_evaluable_rows(target: str, reason: str) -> list[dict[str, object]]:
    return [
        {
            "target": target,
            "model": model,
            "outer_fold": "none",
            "status": "not_evaluable",
            "reason": reason,
            "allocation_authority": 0.0,
        }
        for model in MODEL_FEATURES
    ]


def _not_evaluable_row(
    target: str,
    model: str,
    fold: str,
    reason: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_brier: float,
) -> dict[str, object]:
    return {
        "target": target,
        "model": model,
        "outer_fold": fold,
        "status": "not_evaluable",
        "reason": reason,
        "train_origins": len(train),
        "test_origins": len(test),
        "base_rate_brier_score": base_brier,
        "allocation_authority": 0.0,
    }


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    clean = prices.copy()
    clean.index = pd.to_datetime(clean.index, errors="coerce").tz_localize(None)
    return clean[clean.index.notna()].sort_index().dropna(how="all")


def _truncate_indexed_frame(frame: pd.DataFrame | None, date: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    clean = frame.copy()
    clean.index = pd.to_datetime(clean.index, errors="coerce").tz_localize(None)
    return clean[clean.index.notna()].sort_index().loc[:date]


def _frozen_baseline_at_origin(
    baseline: pd.DataFrame | None,
    origin_date: pd.Timestamp,
    *,
    maximum_age_days: int = 100,
) -> dict[str, object]:
    empty = {
        "frozen_trade_bot_risk_budget": np.nan,
        "frozen_trade_bot_risk_reduction": np.nan,
        "frozen_trade_bot_market_date": pd.NaT,
        "frozen_trade_bot_run_id": "",
    }
    if not isinstance(baseline, pd.DataFrame) or baseline.empty:
        return empty
    clean = baseline.copy()
    clean["market_date"] = pd.to_datetime(clean.get("market_date"), errors="coerce")
    eligible = clean[clean["market_date"].le(origin_date)].sort_values("market_date")
    if eligible.empty:
        return empty
    row = eligible.iloc[-1]
    market_date = pd.Timestamp(row["market_date"])
    if (origin_date - market_date).days > maximum_age_days:
        return empty
    budget = pd.to_numeric(pd.Series([row.get("risk_budget_multiplier")]), errors="coerce").iloc[0]
    return {
        "frozen_trade_bot_risk_budget": budget,
        "frozen_trade_bot_risk_reduction": 1.0 - budget if pd.notna(budget) else np.nan,
        "frozen_trade_bot_market_date": market_date,
        "frozen_trade_bot_run_id": row.get("run_id", ""),
    }


def _build_origin_specs(
    prices: pd.DataFrame,
    manifest: pd.DataFrame | None,
    *,
    step: int,
    min_history_days: int,
    default_ticker: str,
) -> list[tuple[int, str, str, str, str]]:
    if not isinstance(manifest, pd.DataFrame) or manifest.empty:
        return [
            (
                location,
                f"calendar_{prices.index[location].year // 3 * 3}_"
                f"{prices.index[location].year // 3 * 3 + 2}",
                f"calendar_{prices.index[location].year // 3 * 3}_"
                f"{prices.index[location].year // 3 * 3 + 2}",
                default_ticker,
                "calendar_blocks_placeholder",
            )
            for location in range(min_history_days, len(prices), step)
        ]
    specs: list[tuple[int, str, str, str, str]] = []
    for _, row in manifest.iterrows():
        start = pd.to_datetime(row.get("start_date"), errors="coerce")
        end = pd.to_datetime(row.get("end_date"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        locations = [
            location
            for location in range(min_history_days, len(prices))
            if start <= prices.index[location] <= end
        ]
        for location in locations[::step]:
            specs.append(
                (
                    location,
                    str(row.get("episode", "unclassified")),
                    str(row.get("episode_cluster", row.get("episode", "unclassified"))),
                    str(row.get("theme_ticker", default_ticker)),
                    "named_manifest",
                )
            )
    return sorted(specs, key=lambda item: (prices.index[item[0]], item[1]))


def _episode_frame(frame: pd.DataFrame | None, episode: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    if "episode" not in frame:
        return frame
    return frame[frame["episode"].astype(str).eq(episode)].copy()


def _episode_balanced_weights(episodes: pd.Series) -> pd.Series:
    counts = episodes.value_counts(dropna=False)
    weights = episodes.map(lambda value: 1.0 / float(counts.loc[value]))
    return weights * (len(weights) / weights.sum())


def _weighted_brier(
    labels: pd.Series,
    probability: pd.Series | np.ndarray,
    weights: pd.Series,
) -> float:
    y = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=float)
    p = np.asarray(probability, dtype=float)
    w = pd.to_numeric(weights, errors="coerce").fillna(1.0).to_numpy(dtype=float)
    return float(np.average((y - p) ** 2, weights=w))


def _concat_nonempty(*frames: pd.DataFrame) -> pd.DataFrame:
    available = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    return pd.concat(available, ignore_index=True) if available else pd.DataFrame()


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weight_array = pd.to_numeric(weights, errors="coerce").fillna(1.0).to_numpy(dtype=float)
    return float(np.average(value_array, weights=weight_array))


def _numeric_product(left: object, right: object) -> float:
    left_value = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_value = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if pd.isna(left_value) or pd.isna(right_value):
        return np.nan
    return float(left_value * right_value)
