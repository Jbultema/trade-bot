from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from trade_bot.ml.models import (
    SklearnFitConfig,
    feature_importance,
    fit_probability_model,
    predict_probability,
)
from trade_bot.research.future_state_ml import build_future_state_features, label_future_states

TaskKind = Literal[
    "future_state",
    "off_ramp",
    "reentry_repair",
    "sector_rotation",
    "strategy_family_router",
    "churn_filter",
]
MLDiagnosticProfile = Literal["standard", "research"]

STANDARD_MODEL_NAMES = (
    "sk_logit_l2",
    "sk_random_forest",
)
RESEARCH_MODEL_NAMES = (
    "sk_logit_l2",
    "sk_logit_l1",
    "sk_random_forest",
    "sk_extra_trees",
    "sk_gradient_boosting",
    "sk_calibrated_logit",
)
DEFAULT_MODEL_NAMES = STANDARD_MODEL_NAMES
DEFAULT_HORIZONS = (21, 63)
RESEARCH_HORIZONS = (5, 21, 63)
STANDARD_STEP_DAYS = 126
RESEARCH_STEP_DAYS = 42
SECTOR_TICKERS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC")


@dataclass(frozen=True)
class MLTaskSpec:
    name: str
    kind: TaskKind
    horizon_days: int
    positive_class: str | None = None
    min_train_observations: int = 252
    train_window_days: int = 756
    step_days: int = 21


@dataclass(frozen=True)
class MLDiagnosticRun:
    output_dir: Path
    metrics: pd.DataFrame
    latest_probabilities: pd.DataFrame
    feature_importance: pd.DataFrame
    family_importance: pd.DataFrame
    drift: pd.DataFrame
    predictions: pd.DataFrame


def run_ml_diagnostics(
    prices: pd.DataFrame,
    *,
    output_dir: str | Path,
    profile: MLDiagnosticProfile = "standard",
    horizons: tuple[int, ...] | None = None,
    model_names: tuple[str, ...] | None = None,
    step_days: int | None = None,
) -> MLDiagnosticRun:
    resolved_horizons, resolved_models, resolved_step_days = _profile_settings(
        profile,
        horizons=horizons,
        model_names=model_names,
        step_days=step_days,
    )
    settings = {
        "profile": profile,
        "horizons": list(resolved_horizons),
        "model_names": list(resolved_models),
        "step_days": resolved_step_days,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = build_future_state_features(prices)
    tasks = _build_task_specs(prices, resolved_horizons, step_days=resolved_step_days)
    metrics_rows: list[dict[str, object]] = []
    latest_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for task in tasks:
        labels = _labels_for_task(prices, task)
        class_columns = tuple(sorted(labels.dropna().astype(str).unique().tolist()))
        if len(class_columns) < 2:
            continue
        for model_name in resolved_models:
            result = _walk_forward_predictions(
                features,
                labels,
                task,
                model_name=model_name,
                class_columns=class_columns,
            )
            if result.predictions.empty:
                continue
            prediction_frames.append(result.predictions)
            if not result.feature_importance.empty:
                importance_frames.append(result.feature_importance)
            metrics_rows.append(
                _score_predictions(
                    result.predictions,
                    task=task,
                    model_name=model_name,
                    class_columns=class_columns,
                    positive_class=task.positive_class,
                )
            )
            latest_probability = _latest_probability(
                features,
                labels,
                task,
                model_name=model_name,
                class_columns=class_columns,
            )
            latest_rows.append(
                {
                    "task": task.name,
                    "kind": task.kind,
                    "horizon_days": task.horizon_days,
                    "model": model_name,
                    "top_class": latest_probability.idxmax() if not latest_probability.empty else "",
                    "top_probability": float(latest_probability.max()) if not latest_probability.empty else np.nan,
                    **{f"prob_{label}": float(latest_probability.get(label, np.nan)) for label in class_columns},
                }
            )

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["utility_score", "brier_score"], ascending=[False, True]
    ) if metrics_rows else pd.DataFrame()
    latest = pd.DataFrame(latest_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    importance = _combine_importance(importance_frames)
    family_importance = _family_importance(importance)
    drift = _feature_drift(features)

    metrics.to_csv(output / "metrics.csv", index=False)
    latest.to_csv(output / "latest_probabilities.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    family_importance.to_csv(output / "family_importance.csv", index=False)
    drift.to_csv(output / "drift.csv", index=False)
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    _write_summary(output, metrics, latest, family_importance, drift, settings=settings)
    return MLDiagnosticRun(
        output_dir=output,
        metrics=metrics,
        latest_probabilities=latest,
        feature_importance=importance,
        family_importance=family_importance,
        drift=drift,
        predictions=predictions,
    )


@dataclass(frozen=True)
class _PredictionResult:
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def _profile_settings(
    profile: MLDiagnosticProfile,
    *,
    horizons: tuple[int, ...] | None,
    model_names: tuple[str, ...] | None,
    step_days: int | None,
) -> tuple[tuple[int, ...], tuple[str, ...], int]:
    if profile == "research":
        default_horizons = RESEARCH_HORIZONS
        default_models = RESEARCH_MODEL_NAMES
        default_step_days = RESEARCH_STEP_DAYS
    else:
        default_horizons = DEFAULT_HORIZONS
        default_models = DEFAULT_MODEL_NAMES
        default_step_days = STANDARD_STEP_DAYS
    return (
        tuple(horizons or default_horizons),
        tuple(model_names or default_models),
        max(int(step_days or default_step_days), 1),
    )


def _build_task_specs(
    prices: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    step_days: int,
) -> tuple[MLTaskSpec, ...]:
    tasks: list[MLTaskSpec] = []
    for horizon in horizons:
        tasks.append(MLTaskSpec(f"future_state_{horizon}d", "future_state", horizon, step_days=step_days))
        tasks.append(
            MLTaskSpec(f"off_ramp_{horizon}d", "off_ramp", horizon, positive_class="risk_off", step_days=step_days)
        )
        tasks.append(
            MLTaskSpec(
                f"reentry_repair_{horizon}d",
                "reentry_repair",
                horizon,
                positive_class="repair",
                step_days=step_days,
            )
        )
        tasks.append(
            MLTaskSpec(
                f"strategy_family_router_{horizon}d",
                "strategy_family_router",
                horizon,
                step_days=step_days,
            )
        )
        if len(set(SECTOR_TICKERS) & set(prices.columns)) >= 4:
            tasks.append(MLTaskSpec(f"sector_rotation_{horizon}d", "sector_rotation", horizon, step_days=step_days))
    tasks.append(MLTaskSpec("churn_filter_21d", "churn_filter", 21, positive_class="durable", step_days=step_days))
    return tuple(tasks)


def _labels_for_task(prices: pd.DataFrame, task: MLTaskSpec) -> pd.Series:
    if task.kind == "future_state":
        return label_future_states(prices, task.horizon_days)
    if task.kind == "off_ramp":
        return _off_ramp_labels(prices, task.horizon_days)
    if task.kind == "reentry_repair":
        return _reentry_labels(prices, task.horizon_days)
    if task.kind == "sector_rotation":
        return _sector_rotation_labels(prices, task.horizon_days)
    if task.kind == "strategy_family_router":
        return _strategy_family_labels(prices, task.horizon_days)
    if task.kind == "churn_filter":
        return _churn_labels(prices, task.horizon_days)
    return pd.Series(index=prices.index, dtype="object")


def _walk_forward_predictions(
    features: pd.DataFrame,
    labels: pd.Series,
    task: MLTaskSpec,
    *,
    model_name: str,
    class_columns: tuple[str, ...],
) -> _PredictionResult:
    prediction_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    start = task.min_train_observations + task.horizon_days
    for fold_id, position in enumerate(range(start, len(features), task.step_days)):
        actual = labels.iloc[position]
        if pd.isna(actual):
            continue
        train_end = position - task.horizon_days
        train_start = max(0, train_end - task.train_window_days)
        x_train = features.iloc[train_start:train_end]
        y_train = labels.iloc[train_start:train_end].dropna().astype(str)
        x_train = x_train.loc[y_train.index]
        if len(y_train) < task.min_train_observations or y_train.nunique() < 2:
            continue
        fit = fit_probability_model(x_train, y_train, _fit_config(model_name, task))
        probability = predict_probability(fit, features.iloc[position], class_columns)
        prediction_rows.append(
            {
                "task": task.name,
                "kind": task.kind,
                "horizon_days": task.horizon_days,
                "model": model_name,
                "fold_id": fold_id,
                "date": features.index[position],
                "actual": str(actual),
                "predicted": str(probability.idxmax()),
                "max_probability": float(probability.max()),
                **{f"prob_{label}": float(probability.get(label, 0.0)) for label in class_columns},
            }
        )
        importance = feature_importance(fit, class_columns)
        for feature, score in importance.head(40).items():
            importance_rows.append(
                {
                    "task": task.name,
                    "kind": task.kind,
                    "horizon_days": task.horizon_days,
                    "model": model_name,
                    "fold_id": fold_id,
                    "feature": feature,
                    "feature_family": feature_family(feature),
                    "importance": float(score),
                }
            )
    return _PredictionResult(pd.DataFrame(prediction_rows), pd.DataFrame(importance_rows))


def _latest_probability(
    features: pd.DataFrame,
    labels: pd.Series,
    task: MLTaskSpec,
    *,
    model_name: str,
    class_columns: tuple[str, ...],
) -> pd.Series:
    train_end = len(features) - task.horizon_days
    train_start = max(0, train_end - task.train_window_days)
    x_train = features.iloc[train_start:train_end]
    y_train = labels.iloc[train_start:train_end].dropna().astype(str)
    x_train = x_train.loc[y_train.index]
    if len(y_train) < task.min_train_observations or y_train.nunique() < 2:
        return pd.Series(dtype=float)
    fit = fit_probability_model(x_train, y_train, _fit_config(model_name, task))
    return predict_probability(fit, features.iloc[-1], class_columns)


def _fit_config(model_name: str, task: MLTaskSpec) -> SklearnFitConfig:
    return SklearnFitConfig(
        model=model_name,
        random_state=17 + task.horizon_days,
        n_estimators=120 if task.horizon_days <= 21 else 160,
        max_depth=4 if task.kind in {"off_ramp", "reentry_repair"} else 5,
        min_samples_leaf=18 if task.horizon_days <= 21 else 25,
        regularization_c=0.55 if task.kind in {"off_ramp", "reentry_repair"} else 0.75,
    )


def _score_predictions(
    predictions: pd.DataFrame,
    *,
    task: MLTaskSpec,
    model_name: str,
    class_columns: tuple[str, ...],
    positive_class: str | None,
) -> dict[str, object]:
    y_true = predictions["actual"].astype(str)
    y_pred = predictions["predicted"].astype(str)
    probability_columns = [f"prob_{label}" for label in class_columns]
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 1e-9, 1.0)
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-9)
    one_hot = np.zeros_like(probabilities)
    class_index = {label: index for index, label in enumerate(class_columns)}
    for row_index, label in enumerate(y_true):
        if label in class_index:
            one_hot[row_index, class_index[label]] = 1.0
    brier = float(np.mean(((probabilities - one_hot) ** 2).sum(axis=1)))
    try:
        loss = float(log_loss(y_true, probabilities, labels=list(class_columns)))
    except ValueError:
        loss = float("nan")
    accuracy = float(accuracy_score(y_true, y_pred))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    ece = _expected_calibration_error(predictions)
    positive_recall = _positive_recall(y_true, y_pred, positive_class)
    utility = _utility_score(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        brier=brier,
        calibration_error=ece,
        positive_recall=positive_recall,
    )
    return {
        "task": task.name,
        "kind": task.kind,
        "horizon_days": task.horizon_days,
        "model": model_name,
        "observations": len(predictions),
        "classes": ", ".join(class_columns),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "brier_score": brier,
        "log_loss": loss,
        "calibration_error": ece,
        "positive_class": positive_class or "",
        "positive_recall": positive_recall,
        "utility_score": utility,
    }


def _off_ramp_labels(prices: pd.DataFrame, horizon_days: int) -> pd.Series:
    spy = _price(prices, "SPY")
    hyg_lqd = _price(prices, "HYG") / _price(prices, "LQD").replace(0, np.nan)
    vixy = _price(prices, "VIXY")
    scale = float(np.sqrt(max(horizon_days, 1) / 21.0))
    forward_return = spy.shift(-horizon_days) / spy.replace(0, np.nan) - 1.0
    forward_min = spy.shift(-1).iloc[::-1].rolling(horizon_days, min_periods=max(2, horizon_days // 3)).min().iloc[::-1]
    forward_drawdown = forward_min / spy.replace(0, np.nan) - 1.0
    credit_return = hyg_lqd.shift(-horizon_days) / hyg_lqd.replace(0, np.nan) - 1.0
    vol_return = vixy.shift(-horizon_days) / vixy.replace(0, np.nan) - 1.0
    risk_off = (
        (forward_return <= -0.040 * scale)
        | (forward_drawdown <= -0.065 * scale)
        | ((credit_return <= -0.020 * scale) & (vol_return >= 0.050 * scale))
    )
    labels = pd.Series("contained", index=prices.index, dtype="object")
    labels.loc[risk_off.fillna(False)] = "risk_off"
    return labels.where(forward_return.notna() & forward_drawdown.notna())


def _reentry_labels(prices: pd.DataFrame, horizon_days: int) -> pd.Series:
    spy = _price(prices, "SPY")
    returns = spy.pct_change(21)
    drawdown = spy / spy.rolling(252, min_periods=60).max().replace(0, np.nan) - 1.0
    future_return = spy.shift(-horizon_days) / spy.replace(0, np.nan) - 1.0
    future_min = spy.shift(-1).iloc[::-1].rolling(horizon_days, min_periods=max(2, horizon_days // 3)).min().iloc[::-1]
    future_drawdown = future_min / spy.replace(0, np.nan) - 1.0
    setup = (drawdown <= -0.04) | (returns <= -0.04)
    repair = setup & (future_return >= 0.025) & (future_drawdown >= -0.045)
    fail = setup & ~repair
    labels = pd.Series(index=prices.index, dtype="object")
    labels.loc[repair.fillna(False)] = "repair"
    labels.loc[fail.fillna(False)] = "falling_knife"
    return labels.where(future_return.notna() & future_drawdown.notna())


def _sector_rotation_labels(prices: pd.DataFrame, horizon_days: int) -> pd.Series:
    tickers = [ticker for ticker in SECTOR_TICKERS if ticker in prices.columns]
    forward = prices[tickers].ffill().shift(-horizon_days) / prices[tickers].ffill().replace(0, np.nan) - 1.0
    valid = forward.notna().sum(axis=1) >= max(3, len(tickers) // 2)
    labels = pd.Series(index=prices.index, dtype="object")
    labels.loc[valid] = forward.loc[valid].idxmax(axis=1).astype("object")
    return labels


def _strategy_family_labels(prices: pd.DataFrame, horizon_days: int) -> pd.Series:
    spy = _forward_return(_price(prices, "SPY"), horizon_days)
    qqq_rsp = _forward_return(_price(prices, "QQQ") / _price(prices, "RSP").replace(0, np.nan), horizon_days)
    smh_spy = _forward_return(_price(prices, "SMH") / _price(prices, "SPY").replace(0, np.nan), horizon_days)
    hyg_lqd = _forward_return(_price(prices, "HYG") / _price(prices, "LQD").replace(0, np.nan), horizon_days)
    sectors = [ticker for ticker in SECTOR_TICKERS if ticker in prices.columns]
    if sectors:
        sector_forward = prices[sectors].ffill().shift(-horizon_days) / prices[sectors].ffill().replace(0, np.nan) - 1.0
        dispersion = sector_forward.max(axis=1) - sector_forward.median(axis=1)
    else:
        dispersion = pd.Series(0.0, index=prices.index)
    drawdown = _price(prices, "SPY") / _price(prices, "SPY").rolling(252, min_periods=60).max().replace(0, np.nan) - 1.0
    labels = pd.Series("balanced_low_churn", index=prices.index, dtype="object")
    labels.loc[(spy <= -0.035) | (hyg_lqd <= -0.020)] = "defensive_carry"
    labels.loc[(drawdown <= -0.06) & (spy > 0.025) & (hyg_lqd > -0.010)] = "dip_reentry"
    labels.loc[(qqq_rsp > 0.020) & (smh_spy > 0.015) & (spy > 0.0)] = "ai_beta"
    labels.loc[(dispersion > 0.035) & (spy > -0.015)] = "sector_rotation"
    labels.loc[(spy > 0.035) & (hyg_lqd > -0.005) & (qqq_rsp.abs() < 0.035)] = "broad_risk"
    return labels.where(spy.notna())


def _churn_labels(prices: pd.DataFrame, horizon_days: int) -> pd.Series:
    features = build_future_state_features(prices)
    instant = features.apply(_instant_state, axis=1)
    future = instant.shift(-horizon_days)
    labels = pd.Series("not_durable", index=features.index, dtype="object")
    labels.loc[instant.eq(future)] = "durable"
    return labels.where(future.notna())


def _instant_state(row: pd.Series) -> str:
    trend = float(row.get("spy_ret_63", 0.0))
    credit = float(row.get("hyg_lqd_ret_63", 0.0))
    volatility = float(row.get("vixy_ret_21", 0.0))
    breadth = float(row.get("rsp_spy_ret_63", 0.0))
    if trend < -0.04 or credit < -0.025 or volatility > 0.12:
        return "risk_off"
    if trend > 0.03 and credit > -0.01 and breadth > -0.01:
        return "risk_on"
    if trend > 0.0:
        return "transition_up"
    return "transition_down"


def _combine_importance(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    grouped = (
        frame.groupby(["task", "kind", "horizon_days", "model", "feature", "feature_family"], as_index=False)
        .agg(mean_importance=("importance", "mean"), fold_hit_rate=("fold_id", "nunique"))
        .sort_values(["task", "model", "mean_importance"], ascending=[True, True, False])
    )
    return grouped


def _family_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    return (
        importance.groupby(["task", "kind", "horizon_days", "model", "feature_family"], as_index=False)
        .agg(mean_importance=("mean_importance", "sum"), represented_features=("feature", "nunique"))
        .sort_values(["task", "model", "mean_importance"], ascending=[True, True, False])
    )


def _feature_drift(features: pd.DataFrame) -> pd.DataFrame:
    if len(features) < 504:
        return pd.DataFrame()
    reference = features.iloc[-1008:-252] if len(features) >= 1260 else features.iloc[:-252]
    recent = features.iloc[-252:]
    rows = []
    for column in features.columns:
        ref = reference[column].dropna().astype(float)
        cur = recent[column].dropna().astype(float)
        if len(ref) < 60 or len(cur) < 30:
            continue
        std = float(ref.std()) or 1.0
        z_score = (float(cur.mean()) - float(ref.mean())) / std
        rows.append(
            {
                "feature": column,
                "feature_family": feature_family(column),
                "recent_mean": float(cur.mean()),
                "reference_mean": float(ref.mean()),
                "mean_shift_z": z_score,
                "psi": _population_stability_index(ref, cur),
                "drift_score": abs(z_score) + min(_population_stability_index(ref, cur), 5.0),
            }
        )
    return pd.DataFrame(rows).sort_values("drift_score", ascending=False)


def feature_family(feature: str) -> str:
    name = feature.lower()
    if "drawdown" in name:
        return "drawdown"
    if name.startswith(("vix", "vixy")) or "vol" in name:
        return "volatility"
    if name.startswith(("hyg", "lqd")) or "hyg_lqd" in name:
        return "credit"
    if name.startswith(("smh", "qqq")) or "qqq_rsp" in name or "smh_spy" in name:
        return "ai_leadership"
    if "rsp_spy" in name or "iwm_spy" in name or name.startswith(("rsp", "iwm")):
        return "breadth"
    if name.startswith(("tlt", "ief")) or "tlt_spy" in name:
        return "duration_rates"
    if name.startswith(("gld", "uso", "dbc")):
        return "commodities"
    if name.startswith("uup"):
        return "dollar"
    if name.startswith("spy"):
        return "broad_equity"
    return "other"


def _expected_calibration_error(predictions: pd.DataFrame, bins: int = 10) -> float:
    confidence = predictions["max_probability"].astype(float).clip(0.0, 1.0)
    correct = predictions["actual"].astype(str).eq(predictions["predicted"].astype(str)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = confidence.ge(lower) & confidence.lt(upper if upper < 1.0 else upper + 1e-9)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(correct.loc[mask].mean()) - float(confidence.loc[mask].mean()))
    return ece


def _positive_recall(y_true: pd.Series, y_pred: pd.Series, positive_class: str | None) -> float:
    if not positive_class:
        return float("nan")
    positives = y_true.eq(positive_class)
    if not positives.any():
        return float("nan")
    return float(y_pred.loc[positives].eq(positive_class).mean())


def _utility_score(
    *,
    accuracy: float,
    balanced_accuracy: float,
    brier: float,
    calibration_error: float,
    positive_recall: float,
) -> float:
    recall = 0.5 if positive_recall != positive_recall else positive_recall
    return float(
        0.25 * accuracy
        + 0.30 * balanced_accuracy
        + 0.20 * recall
        + 0.15 * (1.0 - min(brier, 1.0))
        + 0.10 * (1.0 - min(calibration_error, 1.0))
    )


def _population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    quantiles = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=quantiles)
    cur_counts, _ = np.histogram(current, bins=quantiles)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _forward_return(series: pd.Series, horizon_days: int) -> pd.Series:
    return series.shift(-horizon_days) / series.replace(0, np.nan) - 1.0


def _price(prices: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker in prices:
        return prices[ticker].ffill().astype(float)
    return pd.Series(1.0, index=prices.index, dtype=float)


def _write_summary(
    output: Path,
    metrics: pd.DataFrame,
    latest: pd.DataFrame,
    family_importance: pd.DataFrame,
    drift: pd.DataFrame,
    *,
    settings: dict[str, object],
) -> None:
    lines = ["# ML Diagnostics", "", f"Generated: {datetime.now(UTC).isoformat()}", ""]
    lines.extend(
        [
            "## Settings",
            "",
            f"- Profile: {settings['profile']}",
            f"- Horizons: {', '.join(str(value) for value in settings['horizons'])} trading days",
            f"- Models: {', '.join(str(value) for value in settings['model_names'])}",
            f"- Walk-forward step: {settings['step_days']} trading days",
            "",
        ]
    )
    if not metrics.empty:
        lines.extend(["## Top Model Tasks", ""])
        for _, row in metrics.head(12).iterrows():
            lines.append(
                f"- {row['task']} / {row['model']}: utility {row['utility_score']:.3f}, "
                f"balanced accuracy {row['balanced_accuracy']:.3f}, Brier {row['brier_score']:.3f}."
            )
        lines.append("")
    if not latest.empty:
        lines.extend(["## Latest Probabilities", ""])
        for _, row in latest.sort_values("top_probability", ascending=False).head(10).iterrows():
            lines.append(
                f"- {row['task']} / {row['model']}: {row['top_class']} "
                f"({row['top_probability']:.1%})."
            )
        lines.append("")
    if not family_importance.empty:
        lines.extend(["## Top Feature Families", ""])
        family_view = family_importance.groupby("feature_family", as_index=False)["mean_importance"].mean()
        family_view = family_view.sort_values("mean_importance", ascending=False).head(8)
        for _, row in family_view.iterrows():
            lines.append(f"- {row['feature_family']}: average importance {row['mean_importance']:.3f}.")
        lines.append("")
    if not drift.empty:
        lines.extend(["## Largest Feature Drift", ""])
        for _, row in drift.head(8).iterrows():
            lines.append(
                f"- {row['feature']} ({row['feature_family']}): z-shift {row['mean_shift_z']:.2f}, "
                f"PSI {row['psi']:.2f}."
            )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
