from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from trade_bot.DEFAULT import (
    DEFAULT_SCENARIO_FRAGILE_UPSIDE_MULTIPLIER,
    DEFAULT_SCENARIO_MAX_MULTIPLIER,
    DEFAULT_SCENARIO_MIN_MULTIPLIER,
    DEFAULT_SCENARIO_RISK_ON_MULTIPLIER,
    DEFAULT_SCENARIO_STRESS_MULTIPLIER,
    DEFAULT_SCENARIO_TRANSITION_MULTIPLIER,
)
from trade_bot.ml.models import (
    SklearnFitConfig,
    fit_probability_model,
    is_sklearn_model,
    predict_probability,
)

STATE_COLUMNS = ("risk_off", "transition", "risk_on_fragile", "risk_on")
STRATEGY_DRAWDOWN_COLUMNS = ("stable", "drawdown")


@dataclass(frozen=True)
class FutureStateModelConfig:
    model: Literal[
        "base_rate",
        "transition",
        "bayesian_base_rate",
        "bayesian_transition",
        "knn",
        "feature_bag_knn",
        "centroid",
        "naive_bayes",
        "bayesian_naive_bayes",
        "ridge_logit",
        "tail_specialist",
        "ensemble",
        "bayesian_ensemble",
        "sk_logit_l2",
        "sk_logit_l1",
        "sk_random_forest",
        "sk_extra_trees",
        "sk_gradient_boosting",
        "sk_calibrated_logit",
        "sk_ensemble",
    ]
    horizon_days: int = 21
    feature_set: Literal["core", "ai", "cross_asset", "all"] = "core"
    train_window_days: int = 756
    min_train_observations: int = 252
    refit_every_days: int = 21
    k_neighbors: int = 80
    bag_count: int = 9
    ridge_alpha: float = 0.15
    ridge_learning_rate: float = 0.08
    ridge_iterations: int = 140
    probability_smoothing: float = 0.08
    dirichlet_prior_strength: float = 8.0
    recency_half_life_days: int = 252
    bayesian_variance_floor: float = 0.25
    bayesian_feature_shrinkage: float = 12.0
    sklearn_n_estimators: int = 140
    sklearn_max_depth: int = 5
    sklearn_min_samples_leaf: int = 20
    sklearn_regularization_c: float = 0.70
    sklearn_random_state: int = 17
    stress_multiplier: float = DEFAULT_SCENARIO_STRESS_MULTIPLIER
    transition_multiplier: float = DEFAULT_SCENARIO_TRANSITION_MULTIPLIER
    fragile_upside_multiplier: float = DEFAULT_SCENARIO_FRAGILE_UPSIDE_MULTIPLIER
    risk_on_multiplier: float = DEFAULT_SCENARIO_RISK_ON_MULTIPLIER
    min_multiplier: float = DEFAULT_SCENARIO_MIN_MULTIPLIER
    max_multiplier: float = DEFAULT_SCENARIO_MAX_MULTIPLIER
    risk_off_activation_probability: float = 0.0
    transition_activation_probability: float = 0.0
    fragile_activation_probability: float = 0.0


@dataclass(frozen=True)
class StrategyDrawdownModelConfig:
    model: Literal[
        "base_rate",
        "sk_logit_l2",
        "sk_logit_l1",
        "sk_random_forest",
        "sk_extra_trees",
        "sk_gradient_boosting",
        "sk_calibrated_logit",
        "sk_ensemble",
    ]
    horizon_days: int = 21
    feature_set: Literal["core", "ai", "cross_asset", "all"] = "ai"
    train_window_days: int = 756
    min_train_observations: int = 252
    refit_every_days: int = 126
    future_drawdown_threshold: float = -0.08
    activation_probability: float = 0.42
    stress_multiplier: float = 0.62
    min_multiplier: float = 0.55
    probability_smoothing: float = 0.08
    sklearn_n_estimators: int = 48
    sklearn_max_depth: int = 4
    sklearn_min_samples_leaf: int = 24
    sklearn_regularization_c: float = 0.70
    sklearn_random_state: int = 29


def build_future_state_features(prices: pd.DataFrame) -> pd.DataFrame:
    filled = prices.ffill().sort_index()
    index = filled.index
    features: dict[str, pd.Series] = {}

    for ticker in ("SPY", "QQQ", "RSP", "IWM", "SMH", "HYG", "LQD", "TLT", "GLD", "USO", "DBC", "UUP", "VIXY"):
        series = _price_series(filled, ticker)
        for window in (21, 63, 126):
            features[f"{ticker.lower()}_ret_{window}"] = _return(series, window)
        if ticker in {"SPY", "QQQ", "HYG", "VIXY"}:
            features[f"{ticker.lower()}_vol_21"] = _daily_returns(series).rolling(21).std() * np.sqrt(252)
            features[f"{ticker.lower()}_vol_63"] = _daily_returns(series).rolling(63).std() * np.sqrt(252)

    for lhs, rhs, name in (
        ("QQQ", "RSP", "qqq_rsp"),
        ("SMH", "SPY", "smh_spy"),
        ("RSP", "SPY", "rsp_spy"),
        ("IWM", "SPY", "iwm_spy"),
        ("HYG", "LQD", "hyg_lqd"),
        ("TLT", "SPY", "tlt_spy"),
        ("GLD", "SPY", "gld_spy"),
        ("USO", "SPY", "uso_spy"),
        ("DBC", "SPY", "dbc_spy"),
        ("UUP", "SPY", "uup_spy"),
        ("VIXY", "SPY", "vixy_spy"),
    ):
        ratio = _price_series(filled, lhs) / _price_series(filled, rhs).replace(0, np.nan)
        for window in (21, 63, 126):
            features[f"{name}_ret_{window}"] = _return(ratio, window)

    spy = _price_series(filled, "SPY")
    qqq = _price_series(filled, "QQQ")
    features["spy_drawdown_63"] = _drawdown(spy, 63)
    features["spy_drawdown_252"] = _drawdown(spy, 252)
    features["qqq_drawdown_63"] = _drawdown(qqq, 63)
    features["qqq_drawdown_252"] = _drawdown(qqq, 252)
    frame = pd.DataFrame(features, index=index).replace([np.inf, -np.inf], np.nan)
    return frame.ffill().fillna(0.0).clip(lower=-5.0, upper=5.0)


def label_future_states(prices: pd.DataFrame, horizon_days: int) -> pd.Series:
    filled = prices.ffill().sort_index()
    scale = float(np.sqrt(max(horizon_days, 1) / 21.0))
    spy = _price_series(filled, "SPY")
    qqq = _price_series(filled, "QQQ")
    rsp = _price_series(filled, "RSP")
    smh = _price_series(filled, "SMH")
    hyg_lqd = _price_series(filled, "HYG") / _price_series(filled, "LQD").replace(0, np.nan)
    vixy = _price_series(filled, "VIXY")

    spy_future = _forward_return(spy, horizon_days)
    qqq_future = _forward_return(qqq, horizon_days)
    rsp_future = _forward_return(rsp, horizon_days)
    smh_future = _forward_return(smh, horizon_days)
    credit_future = _forward_return(hyg_lqd, horizon_days)
    vixy_future = _forward_return(vixy, horizon_days)
    forward_drawdown = _forward_min(spy, horizon_days) / spy.replace(0, np.nan) - 1.0

    risk_off = (
        (spy_future <= -0.045 * scale)
        | (forward_drawdown <= -0.065 * scale)
        | ((credit_future <= -0.025 * scale) & (vixy_future >= 0.05 * scale))
    )
    fragile = (
        ~risk_off
        & (spy_future > 0.0)
        & ((qqq_future - rsp_future) >= 0.025 * scale)
        & ((smh_future - spy_future) >= 0.020 * scale)
    )
    risk_on = (
        ~risk_off
        & ~fragile
        & (spy_future >= 0.035 * scale)
        & (credit_future >= -0.010 * scale)
        & ((rsp_future - spy_future) >= -0.020 * scale)
    )

    labels = pd.Series("transition", index=filled.index, dtype="object")
    labels.loc[risk_on.fillna(False)] = "risk_on"
    labels.loc[fragile.fillna(False)] = "risk_on_fragile"
    labels.loc[risk_off.fillna(False)] = "risk_off"
    valid = spy_future.notna() & forward_drawdown.notna()
    return labels.where(valid)


def build_future_state_probabilities(
    prices: pd.DataFrame,
    config: FutureStateModelConfig,
) -> pd.DataFrame:
    features = build_future_state_features(prices)
    labels = label_future_states(prices, config.horizon_days)
    selected_features = _select_feature_columns(features, config.feature_set)
    if selected_features.empty:
        return _uniform_probabilities(features.index)

    probabilities = []
    last_fit_at = -10**9
    last_predict_at = -10**9
    last_probability: pd.Series | None = None
    cached_fit: object | None = None
    cached_columns: list[str] | None = None
    for position, _timestamp in enumerate(selected_features.index):
        if (
            last_probability is not None
            and position - last_predict_at < max(config.refit_every_days, 1)
        ):
            probabilities.append(last_probability.copy())
            continue
        train_end = position - config.horizon_days
        if train_end <= 0:
            last_probability = _default_probability()
            last_predict_at = position
            probabilities.append(last_probability.copy())
            continue
        train_start = max(0, train_end - config.train_window_days)
        x_train = selected_features.iloc[train_start:train_end]
        y_train = labels.iloc[train_start:train_end]
        valid = y_train.notna()
        x_train = x_train.loc[valid]
        y_train = y_train.loc[valid].astype(str)
        if len(y_train) < config.min_train_observations:
            last_probability = _smoothed_probability(_class_frequency(y_train), config)
            last_predict_at = position
            probabilities.append(last_probability.copy())
            continue

        current = selected_features.iloc[position]
        if _uses_cached_fit(config.model):
            if cached_fit is None or position - last_fit_at >= config.refit_every_days:
                cached_columns = list(x_train.columns)
                cached_fit = _fit_cached_model(x_train, y_train, config)
                last_fit_at = position
            model_probability = _predict_cached_model(
                cached_fit,
                current.reindex(cached_columns or list(x_train.columns)).fillna(0.0),
                config,
            )
        else:
            model_probability = _predict_direct_model(x_train, y_train, current, config)

        base_probability = _class_frequency(y_train)
        last_probability = _blend_probabilities(
            model_probability,
            base_probability,
            config.probability_smoothing,
        )
        last_predict_at = position
        probabilities.append(last_probability.copy())

    frame = pd.DataFrame(probabilities, index=selected_features.index, columns=STATE_COLUMNS)
    return _normalize_probabilities(frame).fillna(_default_probability())


def apply_future_state_position_sizing(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    config: FutureStateModelConfig,
    *,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    probabilities = build_future_state_probabilities(prices, config)
    probabilities = probabilities.reindex(target_weights.index).ffill().fillna(_default_probability())
    risk_off_probability = _activated_probability(
        probabilities["risk_off"],
        config.risk_off_activation_probability,
    )
    transition_probability = _activated_probability(
        probabilities["transition"],
        config.transition_activation_probability,
    )
    fragile_probability = _activated_probability(
        probabilities["risk_on_fragile"],
        config.fragile_activation_probability,
    )
    multiplier = config.risk_on_multiplier
    multiplier -= risk_off_probability * (config.risk_on_multiplier - config.stress_multiplier)
    multiplier -= transition_probability * (
        config.risk_on_multiplier - config.transition_multiplier
    )
    multiplier -= fragile_probability * (
        config.risk_on_multiplier - config.fragile_upside_multiplier
    )
    multiplier = multiplier.clip(lower=config.min_multiplier, upper=config.max_multiplier)

    adjusted = target_weights.copy().astype(float)
    if defensive_ticker and defensive_ticker not in adjusted.columns:
        adjusted[defensive_ticker] = 0.0
    risk_columns = [column for column in adjusted.columns if column != defensive_ticker]
    adjusted.loc[:, risk_columns] = adjusted[risk_columns].mul(multiplier, axis=0)
    if defensive_ticker:
        residual = (1.0 - adjusted.sum(axis=1)).clip(lower=0.0)
        adjusted.loc[:, defensive_ticker] = adjusted[defensive_ticker] + residual
    return adjusted.clip(lower=0.0)


def label_strategy_forward_drawdown(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon_days: int,
    future_drawdown_threshold: float,
) -> pd.Series:
    strategy_returns = _strategy_shadow_returns(target_weights, prices)
    equity = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    forward_drawdown = _forward_min(equity, horizon_days) / equity.replace(0, np.nan) - 1.0
    labels = pd.Series("stable", index=equity.index, dtype="object")
    labels.loc[forward_drawdown <= future_drawdown_threshold] = "drawdown"
    return labels.where(forward_drawdown.notna())


def build_strategy_drawdown_probabilities(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    config: StrategyDrawdownModelConfig,
) -> pd.DataFrame:
    features = build_future_state_features(prices)
    labels = label_strategy_forward_drawdown(
        target_weights,
        prices,
        horizon_days=config.horizon_days,
        future_drawdown_threshold=config.future_drawdown_threshold,
    )
    selected_features = _select_feature_columns(features, config.feature_set)
    if selected_features.empty:
        return _default_strategy_drawdown_probabilities(features.index)

    probabilities = []
    last_fit_at = -10**9
    last_predict_at = -10**9
    last_probability: pd.Series | None = None
    cached_fit: object | None = None
    cached_columns: list[str] | None = None
    for position, _timestamp in enumerate(selected_features.index):
        if (
            last_probability is not None
            and position - last_predict_at < max(config.refit_every_days, 1)
        ):
            probabilities.append(last_probability.copy())
            continue
        train_end = position - config.horizon_days
        if train_end <= 0:
            last_probability = _default_strategy_drawdown_probability()
            last_predict_at = position
            probabilities.append(last_probability.copy())
            continue
        train_start = max(0, train_end - config.train_window_days)
        x_train = selected_features.iloc[train_start:train_end]
        y_train = labels.iloc[train_start:train_end]
        valid = y_train.notna()
        x_train = x_train.loc[valid]
        y_train = y_train.loc[valid].astype(str)
        if len(y_train) < config.min_train_observations:
            last_probability = _blend_strategy_drawdown_probabilities(
                _strategy_drawdown_frequency(y_train),
                _default_strategy_drawdown_probability(),
                config.probability_smoothing,
            )
            last_predict_at = position
            probabilities.append(last_probability.copy())
            continue

        current = selected_features.iloc[position]
        if is_sklearn_model(config.model):
            if cached_fit is None or position - last_fit_at >= config.refit_every_days:
                cached_columns = list(x_train.columns)
                cached_fit = fit_probability_model(
                    x_train,
                    y_train,
                    SklearnFitConfig(
                        model=config.model,
                        random_state=config.sklearn_random_state + config.horizon_days,
                        n_estimators=config.sklearn_n_estimators,
                        max_depth=config.sklearn_max_depth,
                        min_samples_leaf=config.sklearn_min_samples_leaf,
                        regularization_c=config.sklearn_regularization_c,
                    ),
                )
                last_fit_at = position
            model_probability = predict_probability(
                cached_fit if isinstance(cached_fit, dict) else {},
                current.reindex(cached_columns or list(x_train.columns)).fillna(0.0),
                STRATEGY_DRAWDOWN_COLUMNS,
            )
        else:
            model_probability = _strategy_drawdown_frequency(y_train)

        base_probability = _strategy_drawdown_frequency(y_train)
        last_probability = _blend_strategy_drawdown_probabilities(
            model_probability,
            base_probability,
            config.probability_smoothing,
        )
        last_predict_at = position
        probabilities.append(last_probability.copy())

    frame = pd.DataFrame(
        probabilities,
        index=selected_features.index,
        columns=STRATEGY_DRAWDOWN_COLUMNS,
    )
    return _normalize_strategy_drawdown_probabilities(frame).fillna(
        _default_strategy_drawdown_probability()
    )


def apply_strategy_drawdown_position_sizing(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    config: StrategyDrawdownModelConfig,
    *,
    defensive_ticker: str | None,
) -> pd.DataFrame:
    probabilities = build_strategy_drawdown_probabilities(target_weights, prices, config)
    probabilities = probabilities.reindex(target_weights.index).ffill().fillna(
        _default_strategy_drawdown_probability()
    )
    drawdown_probability = _activated_probability(
        probabilities["drawdown"],
        config.activation_probability,
    )
    multiplier = 1.0 - drawdown_probability * (1.0 - config.stress_multiplier)
    multiplier = multiplier.clip(lower=config.min_multiplier, upper=1.0)

    adjusted = target_weights.copy().astype(float)
    if defensive_ticker and defensive_ticker not in adjusted.columns:
        adjusted[defensive_ticker] = 0.0
    risk_columns = [column for column in adjusted.columns if column != defensive_ticker]
    adjusted.loc[:, risk_columns] = adjusted[risk_columns].mul(multiplier, axis=0)
    if defensive_ticker:
        residual = (1.0 - adjusted.sum(axis=1)).clip(lower=0.0)
        adjusted.loc[:, defensive_ticker] = adjusted[defensive_ticker] + residual
    return adjusted.clip(lower=0.0)


def _activated_probability(probability: pd.Series, activation_probability: float) -> pd.Series:
    threshold = float(np.clip(activation_probability, 0.0, 0.95))
    if threshold <= 0.0:
        return probability.clip(lower=0.0, upper=1.0)
    return ((probability - threshold) / (1.0 - threshold)).clip(lower=0.0, upper=1.0)


def _strategy_shadow_returns(target_weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    aligned_prices = prices.ffill().sort_index()
    aligned_weights = target_weights.reindex(aligned_prices.index).ffill().fillna(0.0)
    common_columns = [column for column in aligned_weights.columns if column in aligned_prices.columns]
    if not common_columns:
        return pd.Series(0.0, index=aligned_prices.index, dtype=float)
    asset_returns = aligned_prices[common_columns].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    execution_weights = aligned_weights[common_columns].shift(1).fillna(0.0)
    return (execution_weights * asset_returns).sum(axis=1)


def _strategy_drawdown_frequency(labels: pd.Series) -> pd.Series:
    counts = labels.value_counts(normalize=True)
    return _normalize_strategy_drawdown_probability(
        pd.Series(
            {
                "stable": float(counts.get("stable", 0.0)),
                "drawdown": float(counts.get("drawdown", 0.0)),
            },
            dtype=float,
        )
    )


def _default_strategy_drawdown_probability() -> pd.Series:
    return pd.Series({"stable": 0.80, "drawdown": 0.20}, dtype=float)


def _default_strategy_drawdown_probabilities(index: pd.Index) -> pd.DataFrame:
    probability = _default_strategy_drawdown_probability()
    return pd.DataFrame(
        [probability] * len(index),
        index=index,
        columns=STRATEGY_DRAWDOWN_COLUMNS,
    )


def _blend_strategy_drawdown_probabilities(
    left: pd.Series,
    right: pd.Series,
    right_weight: float,
) -> pd.Series:
    weight = float(np.clip(right_weight, 0.0, 1.0))
    left = left.reindex(STRATEGY_DRAWDOWN_COLUMNS).fillna(0.0)
    right = right.reindex(STRATEGY_DRAWDOWN_COLUMNS).fillna(0.0)
    return _normalize_strategy_drawdown_probability(left * (1.0 - weight) + right * weight)


def _normalize_strategy_drawdown_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.reindex(columns=STRATEGY_DRAWDOWN_COLUMNS).clip(lower=0.0).fillna(0.0)
    totals = normalized.sum(axis=1).replace(0.0, np.nan)
    return normalized.div(totals, axis=0).fillna(_default_strategy_drawdown_probability())


def _normalize_strategy_drawdown_probability(probability: pd.Series) -> pd.Series:
    probability = probability.reindex(STRATEGY_DRAWDOWN_COLUMNS).fillna(0.0).clip(lower=0.0)
    total = float(probability.sum())
    if total <= 0:
        return _default_strategy_drawdown_probability()
    return probability / total


def _predict_direct_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    current: pd.Series,
    config: FutureStateModelConfig,
) -> pd.Series:
    if config.model == "base_rate":
        return _class_frequency(y_train)
    if config.model == "transition":
        return _transition_probability(x_train, y_train, current)
    if config.model == "bayesian_base_rate":
        return _bayesian_base_rate_probability(y_train, config)
    if config.model == "bayesian_transition":
        return _bayesian_transition_probability(x_train, y_train, current, config)
    if config.model == "knn":
        return _knn_probability(x_train, y_train, current, config.k_neighbors)
    if config.model == "feature_bag_knn":
        return _feature_bag_probability(x_train, y_train, current, config)
    if config.model == "centroid":
        return _centroid_probability(x_train, y_train, current)
    if config.model == "naive_bayes":
        return _naive_bayes_probability(x_train, y_train, current)
    if config.model == "bayesian_naive_bayes":
        return _bayesian_naive_bayes_probability(x_train, y_train, current, config)
    if config.model == "tail_specialist":
        return _tail_specialist_probability(x_train, y_train, current, config)
    if config.model == "ensemble":
        pieces = [
            _transition_probability(x_train, y_train, current),
            _knn_probability(x_train, y_train, current, config.k_neighbors),
            _centroid_probability(x_train, y_train, current),
            _naive_bayes_probability(x_train, y_train, current),
        ]
        return _normalize_probability(sum(pieces) / len(pieces))
    if config.model == "bayesian_ensemble":
        pieces = [
            _bayesian_base_rate_probability(y_train, config),
            _bayesian_transition_probability(x_train, y_train, current, config),
            _bayesian_naive_bayes_probability(x_train, y_train, current, config),
            _tail_specialist_probability(x_train, y_train, current, config),
        ]
        weights = (0.15, 0.30, 0.35, 0.20)
        probability = sum(piece * weight for piece, weight in zip(pieces, weights, strict=True))
        return _normalize_probability(probability)
    return _class_frequency(y_train)


def _uses_cached_fit(model: str) -> bool:
    return model == "ridge_logit" or is_sklearn_model(model)


def _fit_cached_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: FutureStateModelConfig,
) -> dict[str, object]:
    if config.model == "ridge_logit":
        return _fit_ridge_logit(x_train, y_train, config)
    if is_sklearn_model(config.model):
        return fit_probability_model(
            x_train,
            y_train,
            SklearnFitConfig(
                model=config.model,
                random_state=config.sklearn_random_state + config.horizon_days,
                n_estimators=config.sklearn_n_estimators,
                max_depth=config.sklearn_max_depth,
                min_samples_leaf=config.sklearn_min_samples_leaf,
                regularization_c=config.sklearn_regularization_c,
            ),
        )
    return {"frequency": _class_frequency(y_train)}


def _predict_cached_model(
    cached_fit: object | None,
    current: pd.Series,
    config: FutureStateModelConfig,
) -> pd.Series:
    if not isinstance(cached_fit, dict):
        return _default_probability()
    if config.model == "ridge_logit":
        return _predict_ridge_logit(cached_fit, current)
    if is_sklearn_model(config.model):
        return predict_probability(cached_fit, current, STATE_COLUMNS)
    frequency = cached_fit.get("frequency")
    return frequency if isinstance(frequency, pd.Series) else _default_probability()


def _fit_ridge_logit(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: FutureStateModelConfig,
) -> dict[str, object]:
    x_scaled, mean, std = _standardize_frame(x_train)
    x_values = np.column_stack([np.ones(len(x_scaled)), x_scaled.to_numpy(dtype=float)])
    y_codes = np.array([STATE_COLUMNS.index(label) for label in y_train], dtype=int)
    y_one_hot = np.eye(len(STATE_COLUMNS))[y_codes]
    weights = np.zeros((x_values.shape[1], len(STATE_COLUMNS)), dtype=float)
    class_counts = np.bincount(y_codes, minlength=len(STATE_COLUMNS)).astype(float)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    row_weights = class_weights[y_codes]
    for _ in range(config.ridge_iterations):
        logits = np.clip(x_values @ weights, -30.0, 30.0)
        probabilities = _softmax_array(logits)
        error = (probabilities - y_one_hot) * row_weights[:, None]
        gradient = x_values.T @ error / max(float(row_weights.sum()), 1.0)
        gradient[1:] += config.ridge_alpha * weights[1:]
        weights -= config.ridge_learning_rate * gradient
    return {"weights": weights, "mean": mean, "std": std, "columns": list(x_train.columns)}


def _predict_ridge_logit(fit: dict[str, object], current: pd.Series) -> pd.Series:
    weights = fit.get("weights")
    mean = fit.get("mean")
    std = fit.get("std")
    columns = fit.get("columns")
    if not isinstance(weights, np.ndarray) or not isinstance(mean, pd.Series) or not isinstance(std, pd.Series):
        return _default_probability()
    if not isinstance(columns, list):
        return _default_probability()
    row = current.reindex(columns).fillna(0.0)
    scaled = ((row - mean) / std.replace(0.0, 1.0)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    values = np.concatenate([[1.0], scaled.to_numpy(dtype=float)])
    probabilities = _softmax_array(np.clip(values @ weights, -30.0, 30.0).reshape(1, -1))[0]
    return _normalize_probability(pd.Series(probabilities, index=STATE_COLUMNS))


def _knn_probability(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    current: pd.Series,
    k_neighbors: int,
) -> pd.Series:
    x_scaled, mean, std = _standardize_frame(x_train)
    current_scaled = ((current.reindex(x_train.columns).fillna(0.0) - mean) / std.replace(0.0, 1.0))
    current_scaled = current_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    distances = np.sqrt(((x_scaled - current_scaled) ** 2).sum(axis=1))
    if distances.empty:
        return _class_frequency(y_train)
    k = min(max(k_neighbors, 5), len(distances))
    nearest = distances.nsmallest(k)
    weights = 1.0 / (nearest + 0.10)
    probability = pd.Series(0.0, index=STATE_COLUMNS)
    for label, weight in zip(y_train.loc[nearest.index], weights, strict=False):
        probability.loc[label] += float(weight)
    return _normalize_probability(probability)


def _feature_bag_probability(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    current: pd.Series,
    config: FutureStateModelConfig,
) -> pd.Series:
    columns = list(x_train.columns)
    if len(columns) <= 4:
        return _knn_probability(x_train, y_train, current, config.k_neighbors)
    rng = np.random.default_rng(17 + config.horizon_days + len(columns))
    probabilities = []
    bag_size = max(4, int(round(len(columns) * 0.55)))
    for _ in range(max(config.bag_count, 3)):
        chosen = sorted(rng.choice(columns, size=bag_size, replace=False).tolist())
        probabilities.append(
            _knn_probability(x_train[chosen], y_train, current.reindex(chosen), config.k_neighbors)
        )
    return _normalize_probability(sum(probabilities) / len(probabilities))


def _centroid_probability(x_train: pd.DataFrame, y_train: pd.Series, current: pd.Series) -> pd.Series:
    x_scaled, mean, std = _standardize_frame(x_train)
    current_scaled = ((current.reindex(x_train.columns).fillna(0.0) - mean) / std.replace(0.0, 1.0))
    current_scaled = current_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    distances = pd.Series(index=STATE_COLUMNS, dtype=float)
    for state in STATE_COLUMNS:
        subset = x_scaled.loc[y_train == state]
        if subset.empty:
            distances.loc[state] = np.nan
            continue
        centroid = subset.mean(axis=0)
        distances.loc[state] = float(np.sqrt(((centroid - current_scaled) ** 2).sum()))
    scores = -distances.fillna(distances.max(skipna=True) + 1.0)
    return _softmax_series(scores)


def _naive_bayes_probability(x_train: pd.DataFrame, y_train: pd.Series, current: pd.Series) -> pd.Series:
    x_scaled, mean, std = _standardize_frame(x_train)
    current_scaled = ((current.reindex(x_train.columns).fillna(0.0) - mean) / std.replace(0.0, 1.0))
    current_scaled = current_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    prior = _class_frequency(y_train).clip(lower=1e-6)
    log_scores = pd.Series(index=STATE_COLUMNS, dtype=float)
    for state in STATE_COLUMNS:
        subset = x_scaled.loc[y_train == state]
        if len(subset) < 5:
            log_scores.loc[state] = np.log(prior.loc[state]) - 50.0
            continue
        state_mean = subset.mean(axis=0)
        state_var = subset.var(axis=0).fillna(1.0).clip(lower=0.20)
        log_likelihood = -0.5 * (((current_scaled - state_mean) ** 2 / state_var) + np.log(state_var)).sum()
        log_scores.loc[state] = float(np.log(prior.loc[state]) + log_likelihood)
    return _softmax_series(log_scores)


def _bayesian_base_rate_probability(
    y_train: pd.Series,
    config: FutureStateModelConfig,
    *,
    prior_probability: pd.Series | None = None,
) -> pd.Series:
    counts = _weighted_class_counts(y_train, config)
    return _dirichlet_posterior(counts, config, prior_probability=prior_probability)


def _bayesian_transition_probability(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    current: pd.Series,
    config: FutureStateModelConfig,
) -> pd.Series:
    base_probability = _bayesian_base_rate_probability(y_train, config)
    current_state = _instant_state(current)
    train_states = x_train.apply(_instant_state, axis=1)
    matching = y_train.loc[train_states == current_state]
    if matching.empty:
        return base_probability
    state_counts = _weighted_class_counts(matching, config)
    min_matches = max(20, int(len(y_train) * 0.06))
    posterior = _dirichlet_posterior(
        state_counts,
        config,
        prior_probability=base_probability,
        effective_prior_strength=(config.dirichlet_prior_strength * (1.5 if len(matching) < min_matches else 1.0)),
    )
    return _blend_probabilities(posterior, base_probability, 0.15 if len(matching) >= min_matches else 0.35)


def _bayesian_naive_bayes_probability(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    current: pd.Series,
    config: FutureStateModelConfig,
) -> pd.Series:
    x_scaled, mean, std = _standardize_frame(x_train)
    current_scaled = ((current.reindex(x_train.columns).fillna(0.0) - mean) / std.replace(0.0, 1.0))
    current_scaled = current_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    prior = _bayesian_base_rate_probability(y_train, config).clip(lower=1e-6)
    global_mean = x_scaled.mean(axis=0).fillna(0.0)
    global_var = x_scaled.var(axis=0).fillna(1.0).clip(lower=config.bayesian_variance_floor)
    shrinkage = max(float(config.bayesian_feature_shrinkage), 1.0)
    log_scores = pd.Series(index=STATE_COLUMNS, dtype=float)
    for state in STATE_COLUMNS:
        subset = x_scaled.loc[y_train == state]
        n_obs = float(len(subset))
        if n_obs < 4:
            log_scores.loc[state] = np.log(prior.loc[state]) - 12.0
            continue
        state_mean = subset.mean(axis=0).fillna(0.0)
        state_var = subset.var(axis=0).fillna(global_var).clip(lower=config.bayesian_variance_floor)
        posterior_mean = ((n_obs * state_mean) + (shrinkage * global_mean)) / (n_obs + shrinkage)
        posterior_var = ((n_obs * state_var) + (shrinkage * global_var)) / (n_obs + shrinkage)
        posterior_var = posterior_var.clip(lower=config.bayesian_variance_floor)
        predictive_var = posterior_var * (1.0 + 1.0 / (n_obs + shrinkage))
        raw_log_likelihood = -0.5 * (
            ((current_scaled - posterior_mean) ** 2 / predictive_var) + np.log(predictive_var)
        ).sum()
        feature_scale = max(float(np.sqrt(len(x_train.columns))), 1.0)
        log_likelihood = raw_log_likelihood / feature_scale
        log_scores.loc[state] = float(np.log(prior.loc[state]) + log_likelihood)
    return _softmax_series(log_scores)


def _tail_specialist_probability(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    current: pd.Series,
    config: FutureStateModelConfig,
) -> pd.Series:
    binary_labels = pd.Series(
        np.where(y_train == "risk_off", "risk_off", "not_risk_off"),
        index=y_train.index,
    )
    x_scaled, mean, std = _standardize_frame(x_train)
    current_scaled = ((current.reindex(x_train.columns).fillna(0.0) - mean) / std.replace(0.0, 1.0))
    current_scaled = current_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    distances = np.sqrt(((x_scaled - current_scaled) ** 2).sum(axis=1))
    k = min(max(config.k_neighbors, 5), len(distances))
    nearest = distances.nsmallest(k)
    weights = 1.0 / (nearest + 0.10)
    risk_weight = float(weights.loc[binary_labels.loc[nearest.index] == "risk_off"].sum())
    risk_off_probability = risk_weight / max(float(weights.sum()), 1e-12)
    rest = _centroid_probability(x_train.loc[y_train != "risk_off"], y_train.loc[y_train != "risk_off"], current)
    rest.loc["risk_off"] = 0.0
    rest = _normalize_probability(rest)
    probability = rest * (1.0 - risk_off_probability)
    probability.loc["risk_off"] = risk_off_probability
    return _normalize_probability(probability)


def _transition_probability(x_train: pd.DataFrame, y_train: pd.Series, current: pd.Series) -> pd.Series:
    current_state = _instant_state(current)
    train_states = x_train.apply(_instant_state, axis=1)
    matching = y_train.loc[train_states == current_state]
    if len(matching) < max(25, int(len(y_train) * 0.08)):
        return _class_frequency(y_train)
    return _blend_probabilities(_class_frequency(matching), _class_frequency(y_train), 0.25)


def _weighted_class_counts(y_train: pd.Series, config: FutureStateModelConfig) -> pd.Series:
    labels = y_train.astype(str)
    weights = _recency_weights(len(labels), config.recency_half_life_days)
    counts = pd.Series(0.0, index=STATE_COLUMNS)
    for label, weight in zip(labels, weights, strict=False):
        if label in counts.index:
            counts.loc[label] += float(weight)
    return counts


def _recency_weights(length: int, half_life_days: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=float)
    half_life = max(float(half_life_days), 1.0)
    age = np.arange(length - 1, -1, -1, dtype=float)
    weights = np.power(0.5, age / half_life)
    return weights / max(float(weights.mean()), 1e-12)


def _dirichlet_posterior(
    counts: pd.Series,
    config: FutureStateModelConfig,
    *,
    prior_probability: pd.Series | None = None,
    effective_prior_strength: float | None = None,
) -> pd.Series:
    prior = _default_probability() if prior_probability is None else prior_probability
    prior = _normalize_probability(prior).reindex(STATE_COLUMNS).fillna(0.0)
    strength = max(
        float(config.dirichlet_prior_strength if effective_prior_strength is None else effective_prior_strength),
        0.01,
    )
    posterior = counts.reindex(STATE_COLUMNS).fillna(0.0).clip(lower=0.0) + prior * strength
    return _normalize_probability(posterior)


def _instant_state(row: pd.Series) -> str:
    trend = float(row.get("spy_ret_63", 0.0))
    breadth = float(row.get("rsp_spy_ret_63", 0.0)) + 0.5 * float(row.get("iwm_spy_ret_63", 0.0))
    credit = float(row.get("hyg_lqd_ret_63", 0.0))
    volatility = float(row.get("vixy_ret_21", 0.0))
    ai = float(row.get("qqq_rsp_ret_63", 0.0)) + float(row.get("smh_spy_ret_63", 0.0))
    if trend < -0.04 or credit < -0.025 or volatility > 0.12:
        return "risk_off"
    if trend > 0.03 and breadth > -0.01 and credit > -0.01:
        return "risk_on"
    if trend > 0.0 and ai > 0.05 and breadth < 0.0:
        return "risk_on_fragile"
    return "transition"


def _select_feature_columns(features: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "all":
        return features.copy()
    patterns = {
        "core": ("spy_", "qqq_", "rsp_", "iwm_", "hyg_", "lqd_", "hyg_lqd", "vixy_", "uup_", "rsp_spy", "iwm_spy"),
        "ai": ("spy_", "qqq_", "rsp_", "smh_", "qqq_rsp", "smh_spy", "hyg_lqd", "vixy_"),
        "cross_asset": ("spy_", "qqq_", "rsp_", "hyg_lqd", "tlt_", "gld_", "uso_", "dbc_", "uup_", "vixy_", "tlt_spy", "gld_spy", "uso_spy", "dbc_spy"),
    }
    selected = [column for column in features.columns if column.startswith(patterns.get(feature_set, patterns["core"]))]
    return features[selected].copy()


def _standardize_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean = frame.mean(axis=0)
    std = frame.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    scaled = ((frame - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return scaled.clip(lower=-6.0, upper=6.0), mean, std


def _price_series(prices: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker in prices:
        return prices[ticker].astype(float)
    return pd.Series(1.0, index=prices.index, dtype=float)


def _daily_returns(series: pd.Series) -> pd.Series:
    return series.astype(float).pct_change().replace([np.inf, -np.inf], np.nan)


def _return(series: pd.Series, window: int) -> pd.Series:
    return series.astype(float).pct_change(window).replace([np.inf, -np.inf], np.nan)


def _forward_return(series: pd.Series, horizon_days: int) -> pd.Series:
    return series.shift(-horizon_days) / series.replace(0, np.nan) - 1.0


def _forward_min(series: pd.Series, horizon_days: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon_days, min_periods=max(2, horizon_days // 3)).min().iloc[::-1]


def _drawdown(series: pd.Series, window: int) -> pd.Series:
    peak = series.rolling(window, min_periods=max(2, window // 3)).max()
    return series / peak.replace(0, np.nan) - 1.0


def _class_frequency(labels: pd.Series) -> pd.Series:
    counts = labels.value_counts(normalize=True)
    return _normalize_probability(pd.Series({state: float(counts.get(state, 0.0)) for state in STATE_COLUMNS}))


def _default_probability() -> pd.Series:
    return pd.Series(
        {"risk_off": 0.20, "transition": 0.40, "risk_on_fragile": 0.15, "risk_on": 0.25},
        dtype=float,
    )


def _uniform_probabilities(index: pd.Index) -> pd.DataFrame:
    probability = _default_probability()
    return pd.DataFrame([probability] * len(index), index=index, columns=STATE_COLUMNS)


def _smoothed_probability(probability: pd.Series, config: FutureStateModelConfig) -> pd.Series:
    return _blend_probabilities(probability, _default_probability(), config.probability_smoothing)


def _blend_probabilities(left: pd.Series, right: pd.Series, right_weight: float) -> pd.Series:
    weight = float(np.clip(right_weight, 0.0, 1.0))
    left = left.reindex(STATE_COLUMNS).fillna(0.0)
    right = right.reindex(STATE_COLUMNS).fillna(0.0)
    return _normalize_probability(left * (1.0 - weight) + right * weight)


def _normalize_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.reindex(columns=STATE_COLUMNS).clip(lower=0.0).fillna(0.0)
    totals = normalized.sum(axis=1).replace(0.0, np.nan)
    return normalized.div(totals, axis=0).fillna(_default_probability())


def _normalize_probability(probability: pd.Series) -> pd.Series:
    probability = probability.reindex(STATE_COLUMNS).fillna(0.0).clip(lower=0.0)
    total = float(probability.sum())
    if total <= 0:
        return _default_probability()
    return probability / total


def _softmax_series(scores: pd.Series) -> pd.Series:
    values = scores.reindex(STATE_COLUMNS).astype(float).fillna(-50.0)
    values = np.clip(values - values.max(), -50.0, 50.0)
    weights = np.exp(values)
    return _normalize_probability(pd.Series(weights, index=STATE_COLUMNS))


def _softmax_array(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values, axis=1, keepdims=True)
    weights = np.exp(np.clip(values, -50.0, 50.0))
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
