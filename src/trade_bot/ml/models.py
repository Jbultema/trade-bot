from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SklearnModelName = Literal[
    "sk_logit_l2",
    "sk_logit_l1",
    "sk_random_forest",
    "sk_extra_trees",
    "sk_gradient_boosting",
    "sk_calibrated_logit",
    "sk_ensemble",
]

SKLEARN_MODEL_NAMES: set[str] = {
    "sk_logit_l2",
    "sk_logit_l1",
    "sk_random_forest",
    "sk_extra_trees",
    "sk_gradient_boosting",
    "sk_calibrated_logit",
    "sk_ensemble",
}


@dataclass(frozen=True)
class SklearnFitConfig:
    model: str
    random_state: int = 17
    n_estimators: int = 160
    max_depth: int = 5
    min_samples_leaf: int = 20
    max_iter: int = 1000
    regularization_c: float = 0.65
    calibration_splits: int = 3


def fit_probability_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: SklearnFitConfig,
) -> dict[str, Any]:
    x_train = _clean_features(x_train)
    y_train = y_train.astype(str)
    classes = tuple(sorted(y_train.dropna().unique().tolist()))
    if len(classes) < 2 or x_train.empty:
        return {"type": "constant", "classes": classes, "probability": _class_frequency(y_train)}
    if config.model == "sk_ensemble":
        pieces = [
            fit_probability_model(x_train, y_train, SklearnFitConfig(**{**config.__dict__, "model": "sk_logit_l2"})),
            fit_probability_model(
                x_train, y_train, SklearnFitConfig(**{**config.__dict__, "model": "sk_random_forest"})
            ),
            fit_probability_model(
                x_train,
                y_train,
                SklearnFitConfig(**{**config.__dict__, "model": "sk_gradient_boosting"}),
            ),
        ]
        return {"type": "ensemble", "pieces": pieces, "classes": classes}

    estimator = _build_estimator(config, y_train)
    try:
        _fit_estimator(estimator, x_train, y_train)
    except ValueError:
        fallback = SklearnFitConfig(**{**config.__dict__, "model": "sk_logit_l2"})
        estimator = _build_estimator(fallback, y_train)
        _fit_estimator(estimator, x_train, y_train)
    return {
        "type": "sklearn",
        "model": config.model,
        "estimator": estimator,
        "columns": list(x_train.columns),
        "classes": tuple(str(label) for label in estimator.classes_),
    }


def predict_probability(
    fit: dict[str, Any],
    current: pd.Series,
    class_columns: tuple[str, ...],
) -> pd.Series:
    if fit.get("type") == "ensemble":
        pieces = [predict_probability(piece, current, class_columns) for piece in fit["pieces"]]
        return _normalize_probability(sum(pieces) / len(pieces), class_columns)
    if fit.get("type") == "constant":
        return _normalize_probability(fit.get("probability", pd.Series(dtype=float)), class_columns)
    estimator = fit.get("estimator")
    columns = fit.get("columns")
    classes = fit.get("classes")
    if estimator is None or not isinstance(columns, list) or not isinstance(classes, tuple):
        return _uniform_probability(class_columns)
    row = current.reindex(columns).astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    probabilities = estimator.predict_proba(pd.DataFrame([row.to_numpy()], columns=columns))[0]
    series = pd.Series(0.0, index=class_columns, dtype=float)
    for label, probability in zip(classes, probabilities, strict=False):
        if label in series.index:
            series.loc[label] = float(probability)
    return _normalize_probability(series, class_columns)


def feature_importance(fit: dict[str, Any], class_columns: tuple[str, ...]) -> pd.Series:
    if fit.get("type") == "ensemble":
        pieces = [feature_importance(piece, class_columns) for piece in fit["pieces"]]
        if not pieces:
            return pd.Series(dtype=float)
        return pd.concat(pieces, axis=1).fillna(0.0).mean(axis=1).sort_values(ascending=False)
    estimator = fit.get("estimator")
    columns = fit.get("columns")
    if estimator is None or not isinstance(columns, list):
        return pd.Series(dtype=float)
    final = estimator.steps[-1][1] if hasattr(estimator, "named_steps") else estimator
    if hasattr(final, "feature_importances_"):
        values = np.asarray(final.feature_importances_, dtype=float)
    elif hasattr(final, "coef_"):
        values = np.abs(np.asarray(final.coef_, dtype=float)).mean(axis=0)
    elif isinstance(final, CalibratedClassifierCV):
        values = _calibrated_logit_importance(final, len(columns))
    else:
        values = np.zeros(len(columns), dtype=float)
    if len(values) != len(columns):
        values = np.resize(values, len(columns))
    series = pd.Series(values, index=columns, dtype=float).clip(lower=0.0)
    total = float(series.sum())
    if total > 0:
        series = series / total
    return series.sort_values(ascending=False)


def is_sklearn_model(model: str) -> bool:
    return model in SKLEARN_MODEL_NAMES


def _build_estimator(config: SklearnFitConfig, y_train: pd.Series) -> Any:
    if config.model == "sk_logit_l1":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=1.0,
                        C=config.regularization_c,
                        max_iter=config.max_iter,
                        class_weight="balanced",
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
    if config.model == "sk_random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=config.random_state,
            n_jobs=1,
        )
    if config.model == "sk_extra_trees":
        return ExtraTreesClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=1,
        )
    if config.model == "sk_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=min(config.n_estimators, 180),
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            random_state=config.random_state,
        )
    if config.model == "sk_calibrated_logit" and _calibration_possible(y_train, config):
        base = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        l1_ratio=0.0,
                        C=config.regularization_c,
                        max_iter=config.max_iter,
                        class_weight="balanced",
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
        return CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=config.calibration_splits),
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    l1_ratio=0.0,
                    C=config.regularization_c,
                    max_iter=config.max_iter,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _fit_estimator(estimator: Any, x_train: pd.DataFrame, y_train: pd.Series) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
        estimator.fit(x_train, y_train)


def _calibration_possible(y_train: pd.Series, config: SklearnFitConfig) -> bool:
    counts = y_train.value_counts()
    return len(counts) >= 2 and int(counts.min()) > config.calibration_splits


def _calibrated_logit_importance(estimator: CalibratedClassifierCV, width: int) -> np.ndarray:
    values = []
    for calibrated in getattr(estimator, "calibrated_classifiers_", []):
        inner = getattr(calibrated, "estimator", None)
        if hasattr(inner, "named_steps"):
            final = inner.steps[-1][1]
            if hasattr(final, "coef_"):
                values.append(np.abs(np.asarray(final.coef_, dtype=float)).mean(axis=0))
    if not values:
        return np.zeros(width, dtype=float)
    return np.asarray(values).mean(axis=0)


def _clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)


def _class_frequency(labels: pd.Series) -> pd.Series:
    counts = labels.value_counts(normalize=True)
    return counts.astype(float)


def _uniform_probability(class_columns: tuple[str, ...]) -> pd.Series:
    if not class_columns:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(class_columns), index=class_columns, dtype=float)


def _normalize_probability(probability: pd.Series, class_columns: tuple[str, ...]) -> pd.Series:
    series = probability.reindex(class_columns).fillna(0.0).clip(lower=0.0).astype(float)
    total = float(series.sum())
    if total <= 0.0:
        return _uniform_probability(class_columns)
    return series / total
