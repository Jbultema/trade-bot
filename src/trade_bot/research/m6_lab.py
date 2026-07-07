from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.covariance import OAS, LedoitWolf

QUINTILE_COLUMNS = ("q1", "q2", "q3", "q4", "q5")
DEFAULT_FORECAST_MODELS = (
    "equal_probability",
    "momentum_quintile",
    "inverse_vol_momentum",
    "covariance_sample",
    "covariance_ledoit_wolf",
    "covariance_oas",
    "gorelli_cv_covariance",
    "tradebot_composite",
)
DEFAULT_PORTFOLIO_CONSTRUCTORS = (
    "equal_weight",
    "forecast_long_short",
    "forecast_long_only_top",
)
DEFAULT_COVARIANCE_ESTIMATORS = ("sample", "ledoit_wolf", "oas", "diagonal")


@dataclass(frozen=True)
class M6Window:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class M6LabConfig:
    name: str
    universe: tuple[str, ...]
    windows: tuple[M6Window, ...]
    train_lookback_days: int = 200
    min_train_observations: int = 60
    simulations: int = 2_000
    random_seed: int = 20260707
    cv_window_count: int = 3
    cv_top_n: int = 2
    forecast_models: tuple[str, ...] = DEFAULT_FORECAST_MODELS
    portfolio_constructors: tuple[str, ...] = DEFAULT_PORTFOLIO_CONSTRUCTORS
    covariance_estimators: tuple[str, ...] = DEFAULT_COVARIANCE_ESTIMATORS


@dataclass(frozen=True)
class M6LabRun:
    forecast_scores: pd.DataFrame
    investment_scores: pd.DataFrame
    model_comparison: pd.DataFrame
    period_diagnostics: pd.DataFrame
    forecasts: pd.DataFrame
    portfolio_weights: pd.DataFrame


def load_m6_lab_config(path: str | Path) -> M6LabConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return m6_lab_config_from_mapping(payload)


def m6_lab_config_from_mapping(payload: dict[str, Any]) -> M6LabConfig:
    settings = dict(payload.get("settings") or {})
    universe = tuple(str(ticker).strip().upper() for ticker in payload.get("universe", ()))
    universe = tuple(ticker for ticker in universe if ticker)
    windows = tuple(_window_from_mapping(row) for row in payload.get("windows", ()))
    if not universe:
        msg = "M6 lab config requires a non-empty universe."
        raise ValueError(msg)
    if not windows:
        msg = "M6 lab config requires at least one evaluation window."
        raise ValueError(msg)
    return M6LabConfig(
        name=str(payload.get("name") or "m6_lab"),
        universe=universe,
        windows=windows,
        train_lookback_days=int(settings.get("train_lookback_days", 200)),
        min_train_observations=int(settings.get("min_train_observations", 60)),
        simulations=int(settings.get("simulations", 2_000)),
        random_seed=int(settings.get("random_seed", 20260707)),
        cv_window_count=int(settings.get("cv_window_count", 3)),
        cv_top_n=int(settings.get("cv_top_n", 2)),
        forecast_models=tuple(settings.get("forecast_models", DEFAULT_FORECAST_MODELS)),
        portfolio_constructors=tuple(
            settings.get("portfolio_constructors", DEFAULT_PORTFOLIO_CONSTRUCTORS)
        ),
        covariance_estimators=tuple(
            settings.get("covariance_estimators", DEFAULT_COVARIANCE_ESTIMATORS)
        ),
    )


def run_m6_lab(
    prices: pd.DataFrame,
    *,
    config: M6LabConfig,
) -> M6LabRun:
    clean_prices = _prepare_prices(prices, config.universe)
    forecast_rows: list[pd.DataFrame] = []
    forecast_score_rows: list[dict[str, object]] = []
    investment_rows: list[dict[str, object]] = []
    weight_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, object]] = []

    for window_index, window in enumerate(config.windows):
        train_prices = _training_prices(clean_prices, window, config=config)
        realized = realized_window_returns(clean_prices, window)
        actual_quintiles = realized_return_quintiles(realized)
        if train_prices.empty or realized.empty or actual_quintiles.empty:
            diagnostics_rows.append(
                _period_diagnostic_row(
                    window=window,
                    available_assets=int(realized.shape[0]),
                    selected_estimators=(),
                    note="insufficient_data",
                )
            )
            continue

        forecasts = _forecasts_for_window(
            train_prices,
            clean_prices,
            window=window,
            window_index=window_index,
            config=config,
        )
        selected_estimators = tuple(
            sorted(
                {
                    str(frame.attrs.get("selected_estimators", ""))
                    for frame in forecasts.values()
                    if frame.attrs.get("selected_estimators")
                }
            )
        )
        diagnostics_rows.append(
            _period_diagnostic_row(
                window=window,
                available_assets=int(realized.shape[0]),
                selected_estimators=selected_estimators,
                note="ok",
            )
        )

        for model_name, forecast in forecasts.items():
            aligned_forecast, aligned_actual, aligned_realized = _align_forecast_actual(
                forecast,
                actual_quintiles,
                realized,
            )
            if aligned_forecast.empty:
                continue
            forecast_score_rows.append(
                {
                    "period": window.label,
                    "model": model_name,
                    "assets": int(aligned_forecast.shape[0]),
                    "rps": ranked_probability_score(aligned_forecast, aligned_actual),
                    "top_quintile_hit_rate": _top_quintile_hit_rate(
                        aligned_forecast,
                        aligned_actual,
                    ),
                    "mean_realized_return": float(aligned_realized.mean()),
                }
            )
            forecast_rows.append(
                _forecast_output_frame(
                    forecast=aligned_forecast,
                    period=window.label,
                    model=model_name,
                )
            )
            for constructor in config.portfolio_constructors:
                weights = portfolio_weights_from_forecast(
                    aligned_forecast,
                    constructor=constructor,
                    realized_returns=aligned_realized,
                )
                if weights.empty:
                    continue
                period_return = float(
                    weights.reindex(aligned_realized.index).fillna(0.0).dot(aligned_realized)
                )
                investment_rows.append(
                    {
                        "period": window.label,
                        "model": model_name,
                        "portfolio": constructor,
                        "period_return": period_return,
                        "gross_exposure": float(weights.abs().sum()),
                        "net_exposure": float(weights.sum()),
                        "long_exposure": float(weights.clip(lower=0.0).sum()),
                        "short_exposure": float(weights.clip(upper=0.0).abs().sum()),
                    }
                )
                weight_rows.append(
                    weights.rename("weight")
                    .reset_index()
                    .rename(columns={"index": "ticker"})
                    .assign(period=window.label, model=model_name, portfolio=constructor)
                )

    forecast_scores = pd.DataFrame(forecast_score_rows)
    investment_scores = pd.DataFrame(investment_rows)
    forecasts_frame = (
        pd.concat(forecast_rows, ignore_index=True) if forecast_rows else _empty_forecasts_frame()
    )
    weights_frame = (
        pd.concat(weight_rows, ignore_index=True)
        if weight_rows
        else _empty_portfolio_weights_frame()
    )
    return M6LabRun(
        forecast_scores=forecast_scores,
        investment_scores=investment_scores,
        model_comparison=summarize_m6_lab(forecast_scores, investment_scores),
        period_diagnostics=pd.DataFrame(diagnostics_rows),
        forecasts=forecasts_frame,
        portfolio_weights=weights_frame,
    )


def realized_window_returns(prices: pd.DataFrame, window: M6Window) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    frame = prices.sort_index()
    start_prices = frame.loc[frame.index < window.start].tail(1)
    end_prices = frame.loc[frame.index <= window.end].tail(1)
    if start_prices.empty or end_prices.empty:
        return pd.Series(dtype=float)
    returns = np.log(end_prices.iloc[0] / start_prices.iloc[0])
    return pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def realized_return_quintiles(returns: pd.Series) -> pd.Series:
    clean = pd.to_numeric(returns, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=int)
    ranks = clean.rank(method="first", ascending=True)
    quintiles = np.ceil(ranks / len(clean) * len(QUINTILE_COLUMNS)).astype(int)
    return quintiles.clip(1, len(QUINTILE_COLUMNS)).rename("actual_quintile")


def ranked_probability_score(forecast: pd.DataFrame, actual_quintiles: pd.Series) -> float:
    aligned = forecast.reindex(actual_quintiles.index)
    aligned = _normalize_forecast_probabilities(aligned)
    actual = actual_quintiles.reindex(aligned.index).astype(int)
    if aligned.empty or actual.empty:
        return float("nan")
    forecast_cdf = aligned[list(QUINTILE_COLUMNS)].cumsum(axis=1).iloc[:, :-1]
    actual_one_hot = pd.DataFrame(0.0, index=aligned.index, columns=QUINTILE_COLUMNS)
    for ticker, quintile in actual.items():
        if 1 <= int(quintile) <= len(QUINTILE_COLUMNS):
            actual_one_hot.loc[ticker, QUINTILE_COLUMNS[int(quintile) - 1]] = 1.0
    actual_cdf = actual_one_hot.cumsum(axis=1).iloc[:, :-1]
    scores = ((forecast_cdf - actual_cdf) ** 2).sum(axis=1) / (len(QUINTILE_COLUMNS) - 1)
    return float(scores.mean())


def forecast_equal_probability(tickers: pd.Index | list[str]) -> pd.DataFrame:
    index = pd.Index(tickers, name="ticker")
    probabilities = pd.DataFrame(0.20, index=index, columns=QUINTILE_COLUMNS)
    return probabilities


def forecast_score_quintiles(scores: pd.Series, *, sharpness: float = 0.95) -> pd.DataFrame:
    clean = pd.to_numeric(scores, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.DataFrame(columns=QUINTILE_COLUMNS)
    predicted = realized_return_quintiles(clean)
    columns = np.arange(1, len(QUINTILE_COLUMNS) + 1, dtype=float)
    rows = []
    for ticker, quintile in predicted.items():
        distances = columns - float(quintile)
        weights = np.exp(-(distances**2) / max(2.0 * sharpness**2, 1e-12))
        probabilities = weights / weights.sum()
        rows.append(pd.Series(probabilities, index=QUINTILE_COLUMNS, name=ticker))
    return pd.DataFrame(rows)


def forecast_covariance_monte_carlo(
    train_prices: pd.DataFrame,
    *,
    estimator: str = "sample",
    simulations: int = 2_000,
    random_seed: int = 0,
    horizon_days: int = 20,
    min_train_observations: int = 60,
) -> pd.DataFrame:
    returns = _training_log_returns(train_prices, min_train_observations=min_train_observations)
    if returns.empty:
        return pd.DataFrame(columns=QUINTILE_COLUMNS)
    mean, covariance = _estimate_return_distribution(returns, estimator=estimator)
    if mean.size == 0 or covariance.size == 0:
        return pd.DataFrame(columns=QUINTILE_COLUMNS)
    horizon = max(int(horizon_days), 1)
    rng = np.random.default_rng(random_seed)
    samples = rng.multivariate_normal(
        mean=mean * horizon,
        cov=_regularized_covariance(covariance * horizon),
        size=max(int(simulations), 1),
        check_valid="ignore",
    )
    counts = pd.DataFrame(0, index=returns.columns, columns=QUINTILE_COLUMNS, dtype=float)
    for row in samples:
        quintiles = realized_return_quintiles(pd.Series(row, index=returns.columns))
        for ticker, quintile in quintiles.items():
            counts.loc[ticker, QUINTILE_COLUMNS[int(quintile) - 1]] += 1.0
    return counts.div(counts.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.20)


def portfolio_weights_from_forecast(
    forecast: pd.DataFrame,
    *,
    constructor: str,
    realized_returns: pd.Series | None = None,
) -> pd.Series:
    probabilities = _normalize_forecast_probabilities(forecast)
    if probabilities.empty:
        return pd.Series(dtype=float)
    if constructor == "equal_weight":
        return pd.Series(1.0 / probabilities.shape[0], index=probabilities.index, name="weight")

    expected_quintile = probabilities.dot(np.arange(1, len(QUINTILE_COLUMNS) + 1, dtype=float))
    centered = expected_quintile - float(expected_quintile.mean())
    if constructor == "forecast_long_short":
        return _normalize_abs_weights(centered)
    if constructor == "forecast_long_only_top":
        cutoff = float(expected_quintile.quantile(0.80))
        raw = expected_quintile.where(expected_quintile >= cutoff, 0.0).clip(lower=0.0)
        return _normalize_long_weights(raw)
    if constructor == "oracle_long_short":
        if realized_returns is None:
            return pd.Series(dtype=float)
        return _normalize_abs_weights(realized_returns - float(realized_returns.mean()))
    msg = f"Unknown M6 portfolio constructor: {constructor}"
    raise ValueError(msg)


def summarize_m6_lab(
    forecast_scores: pd.DataFrame,
    investment_scores: pd.DataFrame,
) -> pd.DataFrame:
    forecast_summary = _forecast_model_summary(forecast_scores)
    investment_summary = _investment_model_summary(investment_scores)
    if forecast_summary.empty and investment_summary.empty:
        return pd.DataFrame()
    merged = forecast_summary.merge(
        investment_summary,
        on=["model", "portfolio"],
        how="outer",
    )
    sort_columns = [column for column in ("mean_rps", "annualized_sharpe") if column in merged]
    ascending = [column == "mean_rps" for column in sort_columns]
    if sort_columns:
        merged = merged.sort_values(sort_columns, ascending=ascending)
    return merged.reset_index(drop=True)


def _window_from_mapping(row: dict[str, Any]) -> M6Window:
    return M6Window(
        label=str(row.get("label") or row.get("period") or row.get("start")),
        start=pd.Timestamp(row["start"]),
        end=pd.Timestamp(row["end"]),
    )


def _prepare_prices(prices: pd.DataFrame, universe: tuple[str, ...]) -> pd.DataFrame:
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    frame = frame.reindex(columns=list(universe))
    return frame.dropna(how="all")


def _training_prices(
    prices: pd.DataFrame,
    window: M6Window,
    *,
    config: M6LabConfig,
) -> pd.DataFrame:
    train = prices.loc[prices.index < window.start].tail(config.train_lookback_days + 1)
    return train.dropna(axis=1, thresh=config.min_train_observations)


def _forecasts_for_window(
    train_prices: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    window: M6Window,
    window_index: int,
    config: M6LabConfig,
) -> dict[str, pd.DataFrame]:
    forecasts: dict[str, pd.DataFrame] = {}
    horizon_days = _window_trading_days(prices, window)
    for model_name in config.forecast_models:
        if model_name == "equal_probability":
            forecasts[model_name] = forecast_equal_probability(train_prices.columns)
        elif model_name == "momentum_quintile":
            forecasts[model_name] = forecast_score_quintiles(_trailing_return_score(train_prices))
        elif model_name == "inverse_vol_momentum":
            forecasts[model_name] = forecast_score_quintiles(
                _inverse_vol_momentum_score(train_prices)
            )
        elif model_name == "tradebot_composite":
            forecasts[model_name] = forecast_score_quintiles(
                _tradebot_composite_score(train_prices)
            )
        elif model_name.startswith("covariance_"):
            estimator = model_name.removeprefix("covariance_")
            forecasts[model_name] = forecast_covariance_monte_carlo(
                train_prices,
                estimator=estimator,
                simulations=config.simulations,
                random_seed=config.random_seed + window_index * 1009 + _estimator_seed(estimator),
                horizon_days=horizon_days,
                min_train_observations=config.min_train_observations,
            )
        elif model_name == "gorelli_cv_covariance":
            forecasts[model_name] = _forecast_gorelli_cv_covariance(
                prices,
                window_index=window_index,
                config=config,
            )
        else:
            msg = f"Unknown M6 forecast model: {model_name}"
            raise ValueError(msg)
    return forecasts


def _forecast_gorelli_cv_covariance(
    prices: pd.DataFrame,
    *,
    window_index: int,
    config: M6LabConfig,
) -> pd.DataFrame:
    window = config.windows[window_index]
    candidates = tuple(config.covariance_estimators)
    cv_scores = _covariance_cv_scores(prices, window_index=window_index, config=config)
    if cv_scores.empty:
        selected = candidates[: max(1, config.cv_top_n)]
    else:
        selected = tuple(
            cv_scores.groupby("estimator")["rps"]
            .mean()
            .sort_values()
            .head(max(1, config.cv_top_n))
            .index
        )
    train_prices = _training_prices(prices, window, config=config)
    horizon_days = _window_trading_days(prices, window)
    frames = [
        forecast_covariance_monte_carlo(
            train_prices,
            estimator=estimator,
            simulations=config.simulations,
            random_seed=config.random_seed + window_index * 1009 + _estimator_seed(estimator) + 77,
            horizon_days=horizon_days,
            min_train_observations=config.min_train_observations,
        )
        for estimator in selected
    ]
    forecast = _average_forecasts(frames)
    forecast.attrs["selected_estimators"] = ",".join(selected)
    return forecast


def _covariance_cv_scores(
    prices: pd.DataFrame,
    *,
    window_index: int,
    config: M6LabConfig,
) -> pd.DataFrame:
    start = max(0, window_index - config.cv_window_count)
    cv_windows = list(enumerate(config.windows[start:window_index], start=start))
    rows: list[dict[str, object]] = []
    for cv_index, cv_window in cv_windows:
        train_prices = _training_prices(prices, cv_window, config=config)
        actual = realized_return_quintiles(realized_window_returns(prices, cv_window))
        if train_prices.empty or actual.empty:
            continue
        horizon_days = _window_trading_days(prices, cv_window)
        for estimator in config.covariance_estimators:
            forecast = forecast_covariance_monte_carlo(
                train_prices,
                estimator=estimator,
                simulations=config.simulations,
                random_seed=config.random_seed + cv_index * 1009 + _estimator_seed(estimator),
                horizon_days=horizon_days,
                min_train_observations=config.min_train_observations,
            )
            aligned_forecast = forecast.reindex(actual.index).dropna(how="all")
            aligned_actual = actual.reindex(aligned_forecast.index).dropna()
            aligned_forecast = aligned_forecast.reindex(aligned_actual.index)
            if aligned_forecast.empty:
                continue
            rows.append(
                {
                    "period": cv_window.label,
                    "estimator": estimator,
                    "rps": ranked_probability_score(aligned_forecast, aligned_actual),
                }
            )
    return pd.DataFrame(rows)


def _training_log_returns(
    train_prices: pd.DataFrame,
    *,
    min_train_observations: int,
) -> pd.DataFrame:
    returns = np.log(train_prices / train_prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=1, thresh=min_train_observations)
    if returns.empty:
        return pd.DataFrame()
    return returns.fillna(0.0)


def _estimate_return_distribution(
    returns: pd.DataFrame,
    *,
    estimator: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = returns.to_numpy(dtype=float)
    mean = returns.mean().to_numpy(dtype=float)
    if estimator == "sample":
        covariance = np.cov(values, rowvar=False)
    elif estimator == "ledoit_wolf":
        covariance = LedoitWolf().fit(values).covariance_
    elif estimator == "oas":
        covariance = OAS().fit(values).covariance_
    elif estimator == "diagonal":
        variance = np.nanvar(values, axis=0, ddof=1)
        covariance = np.diag(np.maximum(variance, 1e-10))
    else:
        msg = f"Unknown covariance estimator: {estimator}"
        raise ValueError(msg)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]])
    return mean, covariance


def _regularized_covariance(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    jitter = max(float(np.trace(matrix)) / max(matrix.shape[0], 1) * 1e-8, 1e-10)
    return matrix + np.eye(matrix.shape[0]) * jitter


def _trailing_return_score(train_prices: pd.DataFrame, lookback: int = 63) -> pd.Series:
    scoped = train_prices.tail(lookback + 1)
    if scoped.shape[0] < 2:
        return pd.Series(dtype=float)
    return np.log(scoped.iloc[-1] / scoped.iloc[0]).replace([np.inf, -np.inf], np.nan).dropna()


def _inverse_vol_momentum_score(train_prices: pd.DataFrame) -> pd.Series:
    momentum = _trailing_return_score(train_prices, lookback=63)
    returns = _training_log_returns(train_prices.tail(64), min_train_observations=20)
    vol = returns.std().replace(0.0, np.nan)
    return (momentum / vol).replace([np.inf, -np.inf], np.nan).dropna()


def _tradebot_composite_score(train_prices: pd.DataFrame) -> pd.Series:
    momentum_63 = _trailing_return_score(train_prices, lookback=63)
    momentum_126 = _trailing_return_score(train_prices, lookback=126)
    returns = _training_log_returns(train_prices.tail(127), min_train_observations=40)
    vol = returns.std() * np.sqrt(252)
    drawdown = _asset_drawdown(train_prices.tail(127))
    components = pd.concat(
        [
            _zscore(momentum_63).rename("momentum_63"),
            _zscore(momentum_126).rename("momentum_126"),
            (-_zscore(vol)).rename("low_vol"),
            _zscore(drawdown).rename("drawdown_resilience"),
        ],
        axis=1,
    )
    return components.mean(axis=1).dropna()


def _asset_drawdown(prices: pd.DataFrame) -> pd.Series:
    frame = prices.dropna(axis=1, how="all")
    if frame.empty:
        return pd.Series(dtype=float)
    drawdown = frame / frame.cummax() - 1.0
    return drawdown.min().dropna()


def _zscore(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return clean
    std = float(clean.std())
    if std <= 0.0 or std != std:
        return clean * 0.0
    return (clean - float(clean.mean())) / std


def _window_trading_days(prices: pd.DataFrame, window: M6Window) -> int:
    scoped = prices.loc[(prices.index > window.start) & (prices.index <= window.end)]
    return max(int(scoped.shape[0]), 1)


def _average_forecasts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [_normalize_forecast_probabilities(frame) for frame in frames if not frame.empty]
    if not valid:
        return pd.DataFrame(columns=QUINTILE_COLUMNS)
    all_index = valid[0].index
    for frame in valid[1:]:
        all_index = all_index.union(frame.index)
    stacked = [frame.reindex(all_index).fillna(0.20) for frame in valid]
    return _normalize_forecast_probabilities(sum(stacked) / len(stacked))


def _normalize_forecast_probabilities(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty:
        return pd.DataFrame(columns=QUINTILE_COLUMNS)
    frame = forecast.reindex(columns=QUINTILE_COLUMNS).astype(float).clip(lower=0.0)
    row_sums = frame.sum(axis=1).replace(0.0, np.nan)
    return frame.div(row_sums, axis=0).fillna(1.0 / len(QUINTILE_COLUMNS))


def _align_forecast_actual(
    forecast: pd.DataFrame,
    actual_quintiles: pd.Series,
    realized: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    aligned_forecast = _normalize_forecast_probabilities(forecast)
    common = aligned_forecast.index.intersection(actual_quintiles.index).intersection(
        realized.index
    )
    return (
        aligned_forecast.reindex(common).dropna(how="all"),
        actual_quintiles.reindex(common).dropna(),
        realized.reindex(common).dropna(),
    )


def _top_quintile_hit_rate(forecast: pd.DataFrame, actual_quintiles: pd.Series) -> float:
    predicted = forecast[list(QUINTILE_COLUMNS)].idxmax(axis=1).str.removeprefix("q").astype(int)
    actual = actual_quintiles.reindex(predicted.index).astype(int)
    if predicted.empty:
        return float("nan")
    return float(predicted.eq(actual).mean())


def _forecast_output_frame(forecast: pd.DataFrame, *, period: str, model: str) -> pd.DataFrame:
    return (
        forecast.reset_index()
        .rename(columns={"index": "ticker"})
        .assign(period=period, model=model)[["period", "model", "ticker", *QUINTILE_COLUMNS]]
    )


def _normalize_abs_weights(raw: pd.Series) -> pd.Series:
    clean = pd.to_numeric(raw, errors="coerce").fillna(0.0)
    denominator = float(clean.abs().sum())
    if denominator <= 0.0:
        return pd.Series(0.0, index=clean.index, name="weight")
    return (clean / denominator).rename("weight")


def _normalize_long_weights(raw: pd.Series) -> pd.Series:
    clean = pd.to_numeric(raw, errors="coerce").clip(lower=0.0).fillna(0.0)
    denominator = float(clean.sum())
    if denominator <= 0.0:
        return pd.Series(0.0, index=clean.index, name="weight")
    return (clean / denominator).rename("weight")


def _forecast_model_summary(forecast_scores: pd.DataFrame) -> pd.DataFrame:
    if forecast_scores.empty:
        return pd.DataFrame(columns=["model", "portfolio"])
    grouped = forecast_scores.groupby("model", as_index=False).agg(
        periods=("period", "nunique"),
        mean_rps=("rps", "mean"),
        median_rps=("rps", "median"),
        best_period_rps=("rps", "min"),
        worst_period_rps=("rps", "max"),
        top_quintile_hit_rate=("top_quintile_hit_rate", "mean"),
    )
    grouped["portfolio"] = "forecast_only"
    return grouped


def _investment_model_summary(investment_scores: pd.DataFrame) -> pd.DataFrame:
    if investment_scores.empty:
        return pd.DataFrame(columns=["model", "portfolio"])
    rows = []
    for (model, portfolio), group in investment_scores.groupby(["model", "portfolio"]):
        returns = pd.to_numeric(group["period_return"], errors="coerce").dropna()
        cumulative = float((1.0 + returns).prod() - 1.0) if not returns.empty else float("nan")
        volatility = float(returns.std()) if len(returns) > 1 else float("nan")
        sharpe = (
            float(returns.mean() / volatility * np.sqrt(12))
            if volatility == volatility and volatility > 0.0
            else float("nan")
        )
        rows.append(
            {
                "model": model,
                "portfolio": portfolio,
                "investment_periods": int(group["period"].nunique()),
                "cumulative_return": cumulative,
                "mean_period_return": float(returns.mean()) if not returns.empty else float("nan"),
                "period_volatility": volatility,
                "annualized_sharpe": sharpe,
                "hit_rate": float((returns > 0.0).mean()) if not returns.empty else float("nan"),
                "max_drawdown": _return_series_max_drawdown(returns),
                "mean_gross_exposure": float(group["gross_exposure"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _return_series_max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns.reset_index(drop=True)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _period_diagnostic_row(
    *,
    window: M6Window,
    available_assets: int,
    selected_estimators: tuple[str, ...],
    note: str,
) -> dict[str, object]:
    return {
        "period": window.label,
        "start": window.start.date().isoformat(),
        "end": window.end.date().isoformat(),
        "available_assets": available_assets,
        "selected_estimators": ";".join(selected_estimators),
        "note": note,
    }


def _empty_forecasts_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["period", "model", "ticker", *QUINTILE_COLUMNS])


def _empty_portfolio_weights_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "weight", "period", "model", "portfolio"])


def _estimator_seed(estimator: str) -> int:
    return sum(ord(character) for character in estimator)
