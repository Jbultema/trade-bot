from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import BotConfig
from trade_bot.features.indicators import daily_returns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.defensive_bias_calibration import (
    CRISIS_WINDOWS,
    ERA_WINDOWS,
    _result_from_execution_weights,
    _return_slice_stats,
    comparable_strategy_family,
)
from trade_bot.research.defensive_correction_search import (
    build_point_in_time_correction_signals,
    effective_defensive_weight_path,
)
from trade_bot.research.strategy_outcome_utility import terminal_wealth_from_cagr

DEFAULT_NATIVE_TIMING_HAZARD_DIR = Path("reports/native_timing_hazard")
BREAK_HORIZON_SESSIONS = 63
BREAK_DRAWDOWN_THRESHOLD = -0.10
DEFENSIVE_TICKER = "BIL"
FOCUS_DRAWDOWN_DAMAGE_BUDGET = 0.01
HARD_DRAWDOWN_LIMIT = -0.30
CRISIS_DRAWDOWN_DAMAGE_BUDGET = 0.015
FAMILY_PRIOR_STRENGTHS = (52.0, 104.0, 208.0)
REGULARIZATION_GRID = (0.05, 0.20, 1.0)

MODEL_ROSTER = (
    "market_core_global",
    "market_augmented_global",
    "family_partial_pool",
)
POLICY_ROSTER = (
    "constant_continuous_existing",
    "constant_mild_continuous_existing",
    "global_continuous_existing",
    "global_mild_continuous_existing",
    "family_continuous_existing",
    "family_confirm_accel_existing",
    "family_confirm_age_existing",
    "family_confirm_age_spy_bridge",
)
OUTER_FOLDS = (
    ("fold_2015_2017", "2015-01-01", "2017-12-31"),
    ("fold_2018_2020", "2018-01-01", "2020-12-31"),
    ("fold_2021_2023", "2021-01-01", "2023-12-31"),
    ("fold_2024_2026", "2024-01-01", "2026-12-31"),
)

CORE_FEATURES = (
    "spy_return_21",
    "spy_return_63",
    "spy_drawdown_252",
    "qqq_return_21",
    "qqq_return_63",
    "qqq_drawdown_252",
    "breadth_return_21",
    "breadth_return_63",
    "credit_return_21",
    "credit_return_63",
    "vixy_return_21",
    "realized_vol_21",
    "break_count",
)
AUGMENTED_FEATURES = CORE_FEATURES + (
    "spy_return_126",
    "qqq_return_126",
    "cross_section_dispersion_21",
    "cross_section_dispersion_63",
    "negative_share_21",
    "negative_share_63",
    "leadership_spread_21",
    "leadership_spread_63",
    "duration_return_21",
    "duration_return_63",
    "realized_vol_acceleration",
)
FAMILY_FEATURES = AUGMENTED_FEATURES + (
    "family_defense",
    "family_defense_delta_5",
    "family_defense_delta_21",
    "family_defense_dispersion",
    "family_concentration",
    "family_turnover_21",
    "family_drawdown",
    "warning_age_weeks",
    "non_deterioration_age_weeks",
)


@dataclass(frozen=True)
class NativeTimingHazardRun:
    market_panel: pd.DataFrame
    family_panel: pd.DataFrame
    oos_predictions: pd.DataFrame
    model_diagnostics: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    leave_crisis_out_diagnostics: pd.DataFrame
    policy_selection: pd.DataFrame
    policy_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    era_metrics: pd.DataFrame
    crisis_metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    execution_sensitivity: pd.DataFrame
    block_bootstrap: pd.DataFrame
    current_read: pd.DataFrame
    shadow_candidate: pd.DataFrame
    promotion_gates: pd.DataFrame
    output_paths: dict[str, Path]


def run_native_timing_hazard_research(
    baseline_run: BaselineRun,
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_NATIVE_TIMING_HAZARD_DIR,
) -> NativeTimingHazardRun:
    strategy_families = {
        name: family
        for name in baseline_run.results
        if (family := comparable_strategy_family(name)) is not None
    }
    if not strategy_families:
        raise ValueError("No comparable dynamic strategy paths are available.")

    prices = baseline_run.prices.sort_index()
    market_panel = build_market_hazard_panel(prices)
    family_panel = build_family_state_panel(
        baseline_run,
        strategy_families,
        market_panel["origin_date"],
    )
    model_panel = family_panel.merge(
        market_panel,
        on="origin_date",
        how="left",
        validate="many_to_one",
    )
    oos_predictions, model_diagnostics, inner_predictions = (
        build_nested_oos_predictions(model_panel)
    )
    threshold_sensitivity = build_threshold_sensitivity(
        prices,
        family_panel,
        baseline_diagnostics=model_diagnostics,
    )
    leave_crisis_out_diagnostics = build_leave_crisis_out_diagnostics(
        model_panel
    )
    current_predictions = build_current_predictions(model_panel)
    policy_selection, policy_results = build_nested_policy_results(
        baseline_run,
        config,
        strategy_families,
        oos_predictions,
        inner_predictions,
    )
    (
        policy_metrics,
        fold_metrics,
        era_metrics,
        crisis_metrics,
        cost_sensitivity,
    ) = evaluate_policy_results(
        baseline_run,
        config,
        strategy_families,
        policy_results,
        oos_predictions,
    )
    execution_sensitivity = build_execution_sensitivity(
        baseline_run,
        config,
        policy_results,
        oos_predictions=oos_predictions,
    )
    block_bootstrap = build_paired_block_bootstrap(
        baseline_run,
        config,
        policy_results,
        oos_predictions=oos_predictions,
    )
    current_read = build_current_policy_read(
        baseline_run,
        strategy_families,
        current_predictions,
        policy_selection,
    )
    proposed_shadow_candidate = build_shadow_candidate(
        config,
        policy_selection=policy_selection,
        policy_metrics=policy_metrics,
        execution_sensitivity=execution_sensitivity,
        block_bootstrap=block_bootstrap,
        current_read=current_read,
    )
    shadow_candidate = freeze_shadow_candidate(
        output_dir,
        proposed_shadow_candidate,
    )
    promotion_gates = build_promotion_gates(
        config,
        model_diagnostics,
        threshold_sensitivity,
        leave_crisis_out_diagnostics,
        policy_metrics,
        fold_metrics,
        crisis_metrics,
        cost_sensitivity,
    )
    frames = {
        "market_panel": market_panel,
        "family_panel": family_panel,
        "oos_predictions": oos_predictions,
        "model_diagnostics": model_diagnostics,
        "threshold_sensitivity": threshold_sensitivity,
        "leave_crisis_out_diagnostics": leave_crisis_out_diagnostics,
        "policy_selection": policy_selection,
        "policy_metrics": policy_metrics,
        "fold_metrics": fold_metrics,
        "era_metrics": era_metrics,
        "crisis_metrics": crisis_metrics,
        "cost_sensitivity": cost_sensitivity,
        "execution_sensitivity": execution_sensitivity,
        "block_bootstrap": block_bootstrap,
        "current_read": current_read,
        "shadow_candidate": shadow_candidate,
        "promotion_gates": promotion_gates,
    }
    paths = write_native_timing_outputs(
        output_dir=output_dir,
        config=config,
        prices=prices,
        frames=frames,
    )
    return NativeTimingHazardRun(
        market_panel=market_panel,
        family_panel=family_panel,
        oos_predictions=oos_predictions,
        model_diagnostics=model_diagnostics,
        threshold_sensitivity=threshold_sensitivity,
        leave_crisis_out_diagnostics=leave_crisis_out_diagnostics,
        policy_selection=policy_selection,
        policy_metrics=policy_metrics,
        fold_metrics=fold_metrics,
        era_metrics=era_metrics,
        crisis_metrics=crisis_metrics,
        cost_sensitivity=cost_sensitivity,
        execution_sensitivity=execution_sensitivity,
        block_bootstrap=block_bootstrap,
        current_read=current_read,
        shadow_candidate=shadow_candidate,
        promotion_gates=promotion_gates,
        output_paths=paths,
    )


def weekly_market_dates(index: pd.Index) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(index)).sort_values().unique()
    if dates.empty:
        return dates
    marker = pd.Series(dates, index=dates)
    weekly = marker.groupby(pd.Grouper(freq="W-WED")).max().dropna()
    return pd.DatetimeIndex(weekly.to_numpy())


def build_market_hazard_panel(
    prices: pd.DataFrame,
    *,
    horizon_sessions: int = BREAK_HORIZON_SESSIONS,
    break_threshold: float = BREAK_DRAWDOWN_THRESHOLD,
    min_history_sessions: int = 504,
) -> pd.DataFrame:
    prices = prices.sort_index().ffill()
    required = {"SPY", "QQQ", "RSP", "HYG", "LQD", "VIXY"}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError("Missing hazard inputs: " + ", ".join(missing))

    spy = prices["SPY"]
    qqq = prices["QQQ"]
    breadth = prices["RSP"] / spy
    credit = prices["HYG"] / prices["LQD"]
    duration = (
        prices["TLT"] / prices["SHY"]
        if {"TLT", "SHY"}.issubset(prices.columns)
        else pd.Series(np.nan, index=prices.index)
    )
    spy_daily = spy.pct_change()
    realized_vol_21 = spy_daily.rolling(21, min_periods=21).std() * np.sqrt(252.0)
    realized_vol_63 = spy_daily.rolling(63, min_periods=42).std() * np.sqrt(252.0)

    raw = pd.DataFrame(index=prices.index)
    for ticker, series in (("spy", spy), ("qqq", qqq)):
        raw[f"{ticker}_return_21"] = series.pct_change(21)
        raw[f"{ticker}_return_63"] = series.pct_change(63)
        raw[f"{ticker}_return_126"] = series.pct_change(126)
        raw[f"{ticker}_drawdown_252"] = (
            series / series.rolling(252, min_periods=126).max() - 1.0
        )
    raw["breadth_return_21"] = breadth.pct_change(21)
    raw["breadth_return_63"] = breadth.pct_change(63)
    raw["credit_return_21"] = credit.pct_change(21)
    raw["credit_return_63"] = credit.pct_change(63)
    raw["vixy_return_21"] = prices["VIXY"].pct_change(21)
    raw["realized_vol_21"] = realized_vol_21
    raw["realized_vol_acceleration"] = realized_vol_21 - realized_vol_63
    raw["duration_return_21"] = duration.pct_change(21)
    raw["duration_return_63"] = duration.pct_change(63)

    cross_columns = [
        ticker
        for ticker in (
            "SPY",
            "QQQ",
            "RSP",
            "SMH",
            "IGV",
            "XLF",
            "XLI",
            "XLV",
            "HYG",
            "TLT",
            "GLD",
        )
        if ticker in prices
    ]
    for horizon in (21, 63):
        cross_returns = prices[cross_columns].pct_change(horizon)
        raw[f"cross_section_dispersion_{horizon}"] = cross_returns.std(
            axis=1,
            skipna=True,
        )
        raw[f"negative_share_{horizon}"] = cross_returns.lt(0.0).mean(axis=1)
        raw[f"leadership_spread_{horizon}"] = (
            cross_returns.max(axis=1, skipna=True)
            - cross_returns.median(axis=1, skipna=True)
        )

    correction = build_point_in_time_correction_signals(prices)
    # correction signals are already prior-close; raw continuous features are not.
    prior = raw.shift(1)
    prior["break_count"] = correction["break_count"]

    weekly = weekly_market_dates(prices.index)
    weekly = weekly[weekly >= prices.index[min_history_sessions]]
    panel = prior.reindex(weekly).copy()
    panel.index.name = "origin_date"
    panel = panel.reset_index()
    labels = build_forward_break_labels(
        prices,
        panel["origin_date"],
        horizon_sessions=horizon_sessions,
        break_threshold=break_threshold,
    )
    panel = panel.merge(labels, on="origin_date", how="left", validate="one_to_one")
    panel = add_episode_cluster_weights(panel)
    return panel


def build_forward_break_labels(
    prices: pd.DataFrame,
    origins: pd.Series | pd.Index,
    *,
    horizon_sessions: int,
    break_threshold: float,
) -> pd.DataFrame:
    positions = {pd.Timestamp(date): position for position, date in enumerate(prices.index)}
    rows: list[dict[str, object]] = []
    for value in pd.to_datetime(origins):
        origin = pd.Timestamp(value)
        position = positions[origin]
        maturity_position = position + horizon_sessions
        if maturity_position >= len(prices):
            rows.append(
                {
                    "origin_date": origin,
                    "maturity_date": pd.NaT,
                    "forward_break": np.nan,
                    "forward_worst_drawdown": np.nan,
                }
            )
            continue
        forward = prices[["SPY", "QQQ"]].iloc[
            position : maturity_position + 1
        ]
        relative = forward / forward.iloc[0] - 1.0
        worst = float(relative.min(axis=0).min())
        rows.append(
            {
                "origin_date": origin,
                "maturity_date": prices.index[maturity_position],
                "forward_break": float(worst <= break_threshold),
                "forward_worst_drawdown": worst,
            }
        )
    return pd.DataFrame(rows)


def add_episode_cluster_weights(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.sort_values("origin_date").copy()
    label = pd.to_numeric(output["forward_break"], errors="coerce")
    matured = label.notna()
    positive = label.eq(1.0)
    positive_group = positive.ne(positive.shift(fill_value=False)).cumsum()
    positive_id = pd.Series("", index=output.index, dtype=object)
    positive_id.loc[positive] = (
        "break_" + positive_group.loc[positive].astype(str)
    )

    negative_id = pd.Series("", index=output.index, dtype=object)
    negative_positions = np.arange(int((matured & ~positive).sum()))
    negative_id.loc[matured & ~positive] = (
        "control_" + pd.Series(negative_positions // 13, dtype=str).to_numpy()
    )
    output["episode_cluster"] = positive_id.where(positive, negative_id)
    cluster_size = output.loc[matured].groupby("episode_cluster")[
        "episode_cluster"
    ].transform("size")
    weights = pd.Series(np.nan, index=output.index, dtype=float)
    weights.loc[matured] = 1.0 / cluster_size
    if weights.loc[matured].notna().any():
        weights.loc[matured] *= float(matured.sum()) / float(
            weights.loc[matured].sum()
        )
    output["episode_weight"] = weights
    return output


def build_family_state_panel(
    baseline_run: BaselineRun,
    strategy_families: dict[str, str],
    origins: pd.Series | pd.Index,
) -> pd.DataFrame:
    origin_index = pd.DatetimeIndex(pd.to_datetime(origins))
    rows: list[pd.DataFrame] = []
    for family in sorted(set(strategy_families.values())):
        names = [
            strategy
            for strategy, strategy_family in strategy_families.items()
            if strategy_family == family
        ]
        defense_rows: list[pd.Series] = []
        concentration_rows: list[pd.Series] = []
        turnover_rows: list[pd.Series] = []
        drawdown_rows: list[pd.Series] = []
        for strategy in names:
            result = baseline_run.results[strategy]
            weights = result.weights.astype(float).clip(lower=0.0)
            defense_rows.append(
                effective_defensive_weight_path(weights).rename(strategy)
            )
            risk = weights.drop(columns=[DEFENSIVE_TICKER], errors="ignore")
            risk_total = risk.sum(axis=1).replace(0.0, np.nan)
            normalized = risk.div(risk_total, axis=0)
            concentration_rows.append(
                normalized.pow(2).sum(axis=1).fillna(0.0).rename(strategy)
            )
            turnover_rows.append(
                result.turnover.rolling(21, min_periods=1).sum().rename(strategy)
            )
            drawdown_rows.append(
                (result.equity / result.equity.cummax() - 1.0).rename(strategy)
            )
        defense_frame = pd.concat(defense_rows, axis=1)
        daily = pd.DataFrame(index=defense_frame.index)
        daily["family_defense"] = defense_frame.median(axis=1)
        daily["family_defense_delta_5"] = daily["family_defense"].diff(5)
        daily["family_defense_delta_21"] = daily["family_defense"].diff(21)
        daily["family_defense_dispersion"] = defense_frame.std(axis=1)
        daily["family_concentration"] = pd.concat(
            concentration_rows,
            axis=1,
        ).median(axis=1)
        daily["family_turnover_21"] = pd.concat(
            turnover_rows,
            axis=1,
        ).median(axis=1)
        daily["family_drawdown"] = pd.concat(
            drawdown_rows,
            axis=1,
        ).median(axis=1)
        weekly = daily.shift(1).reindex(origin_index).ffill()
        warning = weekly["family_defense"].ge(0.30)
        weekly["warning_age_weeks"] = consecutive_true_count(warning)
        rows.append(
            weekly.assign(
                origin_date=origin_index,
                family=family,
                family_strategy_count=len(names),
            ).reset_index(drop=True)
        )
    output = pd.concat(rows, ignore_index=True)
    return output


def consecutive_true_count(values: pd.Series) -> pd.Series:
    clean = values.fillna(False).astype(bool)
    groups = (~clean).cumsum()
    return clean.astype(int).groupby(groups).cumsum().astype(int)


def _add_non_deterioration_age(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.sort_values(["family", "origin_date"]).copy()
    values: list[pd.Series] = []
    for _family, group in output.groupby("family", sort=False):
        stable = (
            group["family_defense"].ge(0.30)
            & pd.to_numeric(group["break_count"], errors="coerce").le(1.0)
            & group["family_defense_delta_5"].fillna(0.0).le(0.025)
        )
        values.append(consecutive_true_count(stable).set_axis(group.index))
    output["non_deterioration_age_weeks"] = pd.concat(values).sort_index()
    return output


def _pipeline(features: tuple[str, ...], *, c_value: float) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(features),
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", transformer),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=2_000,
                ),
            ),
        ]
    )


def _fit_probability_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    *,
    c_value: float,
) -> np.ndarray:
    label = pd.to_numeric(train["forward_break"], errors="coerce")
    if label.nunique() < 2:
        return np.full(len(test), float(label.mean()) if not label.empty else 0.0)
    model = _pipeline(features, c_value=c_value)
    fit_kwargs = {
        "model__sample_weight": pd.to_numeric(
            train["episode_weight"],
            errors="coerce",
        )
        .fillna(1.0)
        .to_numpy()
    }
    model.fit(train[list(features)], label.astype(int), **fit_kwargs)
    return model.predict_proba(test[list(features)])[:, 1]


def _predict_model_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    model_name: str,
    c_value: float,
    prior_strength: float,
) -> np.ndarray:
    features = (
        CORE_FEATURES
        if model_name == "market_core_global"
        else AUGMENTED_FEATURES
    )
    global_train = train.sort_values("family").drop_duplicates("origin_date")
    global_test = test.sort_values("family").drop_duplicates("origin_date")
    global_probability = _fit_probability_model(
        global_train,
        global_test,
        features,
        c_value=c_value,
    )
    global_by_date = pd.Series(
        global_probability,
        index=pd.to_datetime(global_test["origin_date"]),
    )
    aligned_global = pd.to_datetime(test["origin_date"]).map(global_by_date).to_numpy(
        dtype=float
    )
    if model_name != "family_partial_pool":
        return aligned_global

    aligned_global_series = pd.Series(
        aligned_global,
        index=test.index,
        dtype=float,
    )
    output = aligned_global_series.copy()
    for family, family_test in test.groupby("family", sort=False):
        family_train = train[train["family"].eq(family)]
        family_probability = _fit_probability_model(
            family_train,
            family_test,
            FAMILY_FEATURES,
            c_value=c_value,
        )
        unique_episodes = family_train["episode_cluster"].replace("", np.nan).nunique()
        family_weight = unique_episodes / (unique_episodes + prior_strength)
        output.loc[family_test.index] = (
            family_weight * family_probability
            + (1.0 - family_weight)
            * aligned_global_series.loc[family_test.index].to_numpy()
        )
    return output.to_numpy()


def _inner_validation_split(
    train: pd.DataFrame,
    *,
    years: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = pd.Timestamp(train["origin_date"].max())
    validation_start = latest - pd.DateOffset(years=years)
    inner_train = train[
        pd.to_datetime(train["maturity_date"]).lt(validation_start)
    ].copy()
    validation = train[pd.to_datetime(train["origin_date"]).ge(validation_start)].copy()
    if inner_train["forward_break"].nunique() < 2 or validation.empty:
        midpoint = pd.to_datetime(train["origin_date"]).quantile(0.70)
        inner_train = train[
            pd.to_datetime(train["maturity_date"]).lt(midpoint)
        ].copy()
        validation = train[pd.to_datetime(train["origin_date"]).ge(midpoint)].copy()
    return inner_train, validation


def build_nested_oos_predictions(
    model_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _add_non_deterioration_age(model_panel)
    matured = panel[panel["forward_break"].notna()].copy()
    prediction_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    inner_rows: list[pd.DataFrame] = []
    for fold_name, start_text, end_text in OUTER_FOLDS:
        test_start = pd.Timestamp(start_text)
        test_end = pd.Timestamp(end_text)
        train = matured[
            pd.to_datetime(matured["maturity_date"]).lt(test_start)
        ].copy()
        test = matured[
            pd.to_datetime(matured["origin_date"]).between(test_start, test_end)
        ].copy()
        if train.empty or test.empty:
            continue
        inner_train, validation = _inner_validation_split(train)
        fold_output = test[
            [
                "origin_date",
                "maturity_date",
                "family",
                "forward_break",
                "forward_worst_drawdown",
                "episode_cluster",
                "episode_weight",
                "break_count",
                "warning_age_weeks",
                "non_deterioration_age_weeks",
            ]
        ].copy()
        train_base_rate = float(
            np.average(
                pd.to_numeric(train["forward_break"], errors="coerce"),
                weights=pd.to_numeric(
                    train["episode_weight"],
                    errors="coerce",
                ).fillna(1.0),
            )
        )
        fold_output["base_rate_probability"] = train_base_rate
        for model_name in MODEL_ROSTER:
            best: tuple[float, float, float] | None = None
            for c_value in REGULARIZATION_GRID:
                priors = (
                    FAMILY_PRIOR_STRENGTHS
                    if model_name == "family_partial_pool"
                    else (104.0,)
                )
                for prior_strength in priors:
                    probability = _predict_model_variant(
                        inner_train,
                        validation,
                        model_name=model_name,
                        c_value=c_value,
                        prior_strength=prior_strength,
                    )
                    score = _weighted_brier(
                        validation["forward_break"],
                        probability,
                        validation["episode_weight"],
                    )
                    candidate = (score, c_value, prior_strength)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                continue
            _score, c_value, prior_strength = best
            inner_probability = _predict_model_variant(
                inner_train,
                validation,
                model_name=model_name,
                c_value=c_value,
                prior_strength=prior_strength,
            )
            inner_frame = validation[
                [
                    "origin_date",
                    "family",
                    "forward_break",
                    "episode_weight",
                    "break_count",
                    "warning_age_weeks",
                    "non_deterioration_age_weeks",
                ]
            ].copy()
            inner_frame["outer_fold"] = fold_name
            inner_frame["model"] = model_name
            inner_frame["probability"] = inner_probability
            inner_frame["base_rate_probability"] = float(
                np.average(
                    pd.to_numeric(
                        inner_train["forward_break"],
                        errors="coerce",
                    ),
                    weights=pd.to_numeric(
                        inner_train["episode_weight"],
                        errors="coerce",
                    ).fillna(1.0),
                )
            )
            inner_rows.append(inner_frame)

            probability = _predict_model_variant(
                train,
                test,
                model_name=model_name,
                c_value=c_value,
                prior_strength=prior_strength,
            )
            fold_output[f"{model_name}_probability"] = probability
            diagnostic_rows.append(
                _model_diagnostic_row(
                    fold_name=fold_name,
                    model_name=model_name,
                    test=test,
                    probability=probability,
                    c_value=c_value,
                    prior_strength=prior_strength,
                    train=train,
                )
            )
        fold_output["outer_fold"] = fold_name
        prediction_rows.append(fold_output)
    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(diagnostic_rows),
        pd.concat(inner_rows, ignore_index=True),
    )


def build_threshold_sensitivity(
    prices: pd.DataFrame,
    family_panel: pd.DataFrame,
    *,
    baseline_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for threshold in (-0.08, BREAK_DRAWDOWN_THRESHOLD, -0.12):
        if threshold == BREAK_DRAWDOWN_THRESHOLD:
            diagnostics = baseline_diagnostics.copy()
        else:
            market_panel = build_market_hazard_panel(
                prices,
                break_threshold=threshold,
            )
            model_panel = family_panel.merge(
                market_panel,
                on="origin_date",
                how="left",
                validate="many_to_one",
            )
            _predictions, diagnostics, _inner = build_nested_oos_predictions(
                model_panel
            )
        diagnostics = diagnostics.copy()
        diagnostics["break_drawdown_threshold"] = threshold
        rows.append(diagnostics)
    output = pd.concat(rows, ignore_index=True)
    output["brier_improvement_vs_base_rate"] = (
        output["base_rate_brier_score"] - output["brier_score"]
    )
    return output


def build_leave_crisis_out_diagnostics(
    model_panel: pd.DataFrame,
) -> pd.DataFrame:
    panel = _add_non_deterioration_age(model_panel)
    matured = panel[panel["forward_break"].notna()].copy()
    rows: list[dict[str, object]] = []
    for crisis, start_text, end_text in CRISIS_WINDOWS:
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        overlaps = (
            pd.to_datetime(matured["origin_date"]).le(end)
            & pd.to_datetime(matured["maturity_date"]).ge(start)
        )
        test = matured[overlaps].copy()
        if test.empty:
            continue
        held_clusters = set(test["episode_cluster"].dropna().astype(str))
        train = matured[
            ~matured["episode_cluster"].astype(str).isin(held_clusters)
        ].copy()
        if train.empty or train["forward_break"].nunique() < 2:
            continue
        for model_name in MODEL_ROSTER:
            probability = _predict_model_variant(
                train,
                test,
                model_name=model_name,
                c_value=0.20,
                prior_strength=104.0,
            )
            diagnostic = _model_diagnostic_row(
                fold_name=f"leave_{crisis}_out",
                model_name=model_name,
                test=test,
                probability=probability,
                c_value=0.20,
                prior_strength=104.0,
                train=train,
            )
            diagnostic.update(
                {
                    "crisis": crisis,
                    "held_out_clusters": len(held_clusters),
                    "brier_improvement_vs_base_rate": float(
                        diagnostic["base_rate_brier_score"]
                    )
                    - float(diagnostic["brier_score"]),
                    "validation_design": "leave_overlapping_episode_clusters_out",
                }
            )
            rows.append(diagnostic)
    return pd.DataFrame(rows)


def build_current_predictions(model_panel: pd.DataFrame) -> pd.DataFrame:
    panel = _add_non_deterioration_age(model_panel)
    matured = panel[panel["forward_break"].notna()].copy()
    current_date = pd.Timestamp(panel["origin_date"].max())
    current = panel[pd.to_datetime(panel["origin_date"]).eq(current_date)].copy()
    inner_train, validation = _inner_validation_split(matured)
    output = current[
        [
            "origin_date",
            "family",
            "break_count",
            "warning_age_weeks",
            "non_deterioration_age_weeks",
        ]
    ].copy()
    output["base_rate_probability"] = float(
        np.average(
            pd.to_numeric(matured["forward_break"], errors="coerce"),
            weights=pd.to_numeric(
                matured["episode_weight"],
                errors="coerce",
            ).fillna(1.0),
        )
    )
    for model_name in MODEL_ROSTER:
        best: tuple[float, float, float] | None = None
        for c_value in REGULARIZATION_GRID:
            priors = (
                FAMILY_PRIOR_STRENGTHS
                if model_name == "family_partial_pool"
                else (104.0,)
            )
            for prior_strength in priors:
                probability = _predict_model_variant(
                    inner_train,
                    validation,
                    model_name=model_name,
                    c_value=c_value,
                    prior_strength=prior_strength,
                )
                candidate = (
                    _weighted_brier(
                        validation["forward_break"],
                        probability,
                        validation["episode_weight"],
                    ),
                    c_value,
                    prior_strength,
                )
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            continue
        _score, c_value, prior_strength = best
        output[f"{model_name}_probability"] = _predict_model_variant(
            matured,
            current,
            model_name=model_name,
            c_value=c_value,
            prior_strength=prior_strength,
        )
    return output


def _weighted_brier(
    label: pd.Series,
    probability: np.ndarray | pd.Series,
    weight: pd.Series,
) -> float:
    return float(
        np.average(
            (pd.to_numeric(label, errors="coerce").to_numpy(dtype=float)
            - np.asarray(probability, dtype=float))
            ** 2,
            weights=pd.to_numeric(weight, errors="coerce").fillna(1.0),
        )
    )


def _model_diagnostic_row(
    *,
    fold_name: str,
    model_name: str,
    test: pd.DataFrame,
    probability: np.ndarray,
    c_value: float,
    prior_strength: float,
    train: pd.DataFrame,
) -> dict[str, object]:
    label = pd.to_numeric(test["forward_break"], errors="coerce").astype(int)
    weight = pd.to_numeric(test["episode_weight"], errors="coerce").fillna(1.0)
    base_probability = float(
        np.average(
            pd.to_numeric(train["forward_break"], errors="coerce"),
            weights=pd.to_numeric(
                train["episode_weight"],
                errors="coerce",
            ).fillna(1.0),
        )
    )
    calibration_bins = pd.qcut(
        pd.Series(probability).rank(method="first"),
        q=min(5, len(probability)),
        labels=False,
        duplicates="drop",
    )
    calibration = pd.DataFrame(
        {
            "label": label.to_numpy(),
            "probability": probability,
            "bin": calibration_bins,
            "weight": weight.to_numpy(),
        }
    )
    calibration_error = 0.0
    total_weight = float(calibration["weight"].sum())
    for _bin, group in calibration.groupby("bin"):
        group_weight = float(group["weight"].sum())
        calibration_error += (
            group_weight
            / total_weight
            * abs(
                float(np.average(group["probability"], weights=group["weight"]))
                - float(np.average(group["label"], weights=group["weight"]))
            )
        )
    return {
        "outer_fold": fold_name,
        "model": model_name,
        "test_rows": len(test),
        "test_unique_dates": test["origin_date"].nunique(),
        "test_event_clusters": test.loc[
            label.eq(1).to_numpy(),
            "episode_cluster",
        ].nunique(),
        "c_value": c_value,
        "family_prior_strength": prior_strength,
        "brier_score": _weighted_brier(label, probability, weight),
        "base_rate_brier_score": _weighted_brier(
            label,
            np.full(len(label), base_probability),
            weight,
        ),
        "log_loss": float(
            log_loss(
                label,
                np.clip(probability, 1e-6, 1.0 - 1e-6),
                sample_weight=weight,
                labels=[0, 1],
            )
        ),
        "roc_auc": (
            float(roc_auc_score(label, probability, sample_weight=weight))
            if label.nunique() > 1
            else np.nan
        ),
        "calibration_error": calibration_error,
    }


def policy_parameter_grid(policy: str) -> list[dict[str, float]]:
    if "mild_continuous" in policy:
        floors = (0.00, 0.05)
        slopes = (0.60, 0.90)
        ceilings = (0.65, 0.80)
        blends = (0.75, 0.90, 0.95)
    else:
        floors = (0.10, 0.20)
        slopes = (1.20, 1.60)
        ceilings = (0.75, 0.90)
        blends = (0.25, 0.50, 0.75)
    accel_values = (
        (0.05, 0.10)
        if "confirm" in policy
        else (0.0,)
    )
    decay_values = (
        (0.01, 0.02)
        if "age" in policy
        else (0.0,)
    )
    rows: list[dict[str, float]] = []
    for floor, slope, ceiling, blend, accel, decay in product(
        floors,
        slopes,
        ceilings,
        blends,
        accel_values,
        decay_values,
    ):
        rows.append(
            {
                "defense_floor": floor,
                "hazard_slope": slope,
                "defense_ceiling": ceiling,
                "native_blend": blend,
                "break_acceleration": accel,
                "age_decay": decay,
            }
        )
    return rows


def continuous_defense_target(
    probability: pd.Series,
    base_defense: pd.Series,
    break_count: pd.Series,
    non_deterioration_age: pd.Series,
    *,
    policy: str,
    parameters: dict[str, float],
) -> pd.Series:
    target = (
        float(parameters["defense_floor"])
        + float(parameters["hazard_slope"]) * probability.astype(float)
    )
    if "confirm" in policy:
        target = target + float(parameters["break_acceleration"]) * break_count.clip(
            0.0,
            4.0,
        )
    if "age" in policy:
        stale_weeks = (non_deterioration_age - 4.0).clip(lower=0.0)
        decay = (
            float(parameters["age_decay"]) * stale_weeks
        ).clip(upper=0.15)
        target = target - decay.where(break_count.le(1.0), 0.0)
    target = target.clip(
        lower=float(parameters["defense_floor"]),
        upper=float(parameters["defense_ceiling"]),
    )
    blend = float(parameters["native_blend"])
    return (blend * base_defense + (1.0 - blend) * target).clip(0.0, 1.0)


def apply_continuous_defense_budget(
    base_weights: pd.DataFrame,
    target_defense: pd.Series,
    *,
    spy_bridge: bool,
    defensive_ticker: str = DEFENSIVE_TICKER,
    max_asset_weight: float = 0.35,
) -> pd.DataFrame:
    adjusted = base_weights.astype(float).clip(lower=0.0).copy()
    if defensive_ticker not in adjusted:
        adjusted[defensive_ticker] = 0.0
    if spy_bridge and "SPY" not in adjusted:
        adjusted["SPY"] = 0.0
    row_sum = adjusted.sum(axis=1)
    over = row_sum.gt(1.0)
    if over.any():
        adjusted.loc[over] = adjusted.loc[over].div(row_sum.loc[over], axis=0)
    risk_columns = [
        column for column in adjusted.columns if column != defensive_ticker
    ]
    target_risk = (
        1.0 - target_defense.reindex(adjusted.index).astype(float)
    ).clip(0.0, 1.0)
    risk = adjusted[risk_columns]
    current_total = risk.sum(axis=1)
    scale = target_risk.div(current_total.where(current_total.gt(1e-12)))
    risk = risk.mul(scale, axis=0).clip(upper=max_asset_weight).fillna(0.0)
    if spy_bridge and "SPY" in risk:
        actual_risk = risk.sum(axis=1)
        addition = pd.concat(
            [
                (target_risk - actual_risk).clip(lower=0.0),
                (max_asset_weight - risk["SPY"]).clip(lower=0.0),
            ],
            axis=1,
        ).min(axis=1)
        risk["SPY"] = risk["SPY"] + addition
    adjusted.loc[:, risk_columns] = risk
    adjusted[defensive_ticker] = (1.0 - risk.sum(axis=1)).clip(0.0, 1.0)
    return adjusted


def _policy_probability_column(policy: str) -> str:
    if policy.startswith("constant_"):
        return "base_rate_probability"
    return (
        "market_augmented_global_probability"
        if policy.startswith("global_")
        else "family_partial_pool_probability"
    )


def _prediction_daily_state(
    predictions: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    policy: str,
) -> pd.DataFrame:
    probability_column = _policy_probability_column(policy)
    weekly = predictions.set_index("origin_date")[
        [
            probability_column,
            "break_count",
            "non_deterioration_age_weeks",
        ]
    ].sort_index()
    daily = weekly.reindex(index).ffill()
    return daily.rename(columns={probability_column: "probability"})


def build_nested_policy_results(
    baseline_run: BaselineRun,
    config: BotConfig,
    strategy_families: dict[str, str],
    oos_predictions: pd.DataFrame,
    inner_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], BacktestResult]]:
    prices = baseline_run.prices.sort_index()
    selections: list[dict[str, object]] = []
    result_weights: dict[tuple[str, str], pd.DataFrame] = {
        (strategy, policy): baseline_run.results[strategy].weights.copy()
        for strategy in strategy_families
        for policy in POLICY_ROSTER
    }
    for fold_name, start_text, end_text in OUTER_FOLDS:
        start = pd.Timestamp(start_text)
        end = min(pd.Timestamp(end_text), prices.index.max())
        fold_oos = oos_predictions[oos_predictions["outer_fold"].eq(fold_name)]
        if fold_oos.empty:
            continue
        for family in sorted(set(strategy_families.values())):
            family_strategies = [
                strategy
                for strategy, strategy_family in strategy_families.items()
                if strategy_family == family
            ]
            for policy in POLICY_ROSTER:
                model_name = (
                    "market_augmented_global"
                    if policy
                    in {
                        "constant_continuous_existing",
                        "constant_mild_continuous_existing",
                        "global_continuous_existing",
                        "global_mild_continuous_existing",
                    }
                    else "family_partial_pool"
                )
                inner = inner_predictions[
                    inner_predictions["outer_fold"].eq(fold_name)
                    & inner_predictions["family"].eq(family)
                    & inner_predictions["model"].eq(model_name)
                ].copy()
                if inner.empty:
                    continue
                wide_inner = inner.rename(
                    columns={"probability": f"{model_name}_probability"}
                )
                best_parameters, best_score = select_policy_parameters(
                    baseline_run,
                    config,
                    family_strategies,
                    wide_inner,
                    policy=policy,
                )
                selections.append(
                    {
                        "outer_fold": fold_name,
                        "family": family,
                        "policy": policy,
                        "inner_utility_score": best_score,
                        **best_parameters,
                    }
                )
                test_prediction = fold_oos[fold_oos["family"].eq(family)].copy()
                prediction_start = pd.Timestamp(
                    test_prediction["origin_date"].min()
                )
                prediction_end = pd.Timestamp(
                    test_prediction["origin_date"].max()
                )
                test_index = prices.index[
                    prices.index.to_series()
                    .between(max(start, prediction_start), min(end, prediction_end))
                    .to_numpy()
                ]
                state = _prediction_daily_state(
                    test_prediction,
                    test_index,
                    policy=policy,
                )
                for strategy in family_strategies:
                    base_weights = baseline_run.results[strategy].weights.loc[test_index]
                    base_defense = effective_defensive_weight_path(base_weights)
                    target = continuous_defense_target(
                        state["probability"],
                        base_defense,
                        state["break_count"],
                        state["non_deterioration_age_weeks"],
                        policy=policy,
                        parameters=best_parameters,
                    )
                    adjusted = apply_continuous_defense_budget(
                        base_weights,
                        target,
                        spy_bridge=policy.endswith("spy_bridge"),
                    )
                    destination = result_weights[(strategy, policy)]
                    for column in adjusted.columns.difference(destination.columns):
                        destination[column] = 0.0
                    destination.loc[test_index, adjusted.columns] = adjusted
    results = {
        (strategy, policy): _result_from_execution_weights(
            baseline_run.results[strategy],
            prices,
            weights,
            transaction_cost_bps=float(config.execution.transaction_cost_bps),
            name=f"{strategy}__{policy}",
        )
        for (strategy, policy), weights in result_weights.items()
    }
    return pd.DataFrame(selections), results


def select_policy_parameters(
    baseline_run: BaselineRun,
    config: BotConfig,
    strategies: list[str],
    predictions: pd.DataFrame,
    *,
    policy: str,
) -> tuple[dict[str, float], float]:
    probability_column = _policy_probability_column(policy)
    if probability_column not in predictions:
        source = (
            "market_augmented_global_probability"
            if probability_column.startswith("market_")
            else "family_partial_pool_probability"
        )
        predictions[probability_column] = predictions[source]
    start = pd.Timestamp(predictions["origin_date"].min())
    end = pd.Timestamp(predictions["origin_date"].max())
    index = baseline_run.prices.index[
        baseline_run.prices.index.to_series().between(start, end).to_numpy()
    ]
    state = _prediction_daily_state(predictions, index, policy=policy)
    best_parameters: dict[str, float] | None = None
    best_score = -np.inf
    spy_return = daily_returns(baseline_run.prices)["SPY"].reindex(index).fillna(0.0)
    for parameters in policy_parameter_grid(policy):
        cagr_deltas: list[float] = []
        drawdown_deltas: list[float] = []
        upside_deltas: list[float] = []
        feasible = True
        for strategy in strategies:
            base = baseline_run.results[strategy]
            base_weights = base.weights.loc[index]
            base_defense = effective_defensive_weight_path(base_weights)
            target = continuous_defense_target(
                state["probability"],
                base_defense,
                state["break_count"],
                state["non_deterioration_age_weeks"],
                policy=policy,
                parameters=parameters,
            )
            adjusted = apply_continuous_defense_budget(
                base_weights,
                target,
                spy_bridge=policy.endswith("spy_bridge"),
            )
            candidate = _result_from_execution_weights(
                base,
                baseline_run.prices,
                adjusted,
                transaction_cost_bps=float(config.execution.transaction_cost_bps),
                name=f"{strategy}__inner",
            )
            base_stats = _return_slice_stats(base.returns, start, end)
            candidate_stats = _return_slice_stats(candidate.returns, start, end)
            cagr_delta = float(candidate_stats["annualized_return"]) - float(
                base_stats["annualized_return"]
            )
            drawdown_delta = float(candidate_stats["max_drawdown"]) - float(
                base_stats["max_drawdown"]
            )
            cagr_deltas.append(cagr_delta)
            drawdown_deltas.append(drawdown_delta)
            upside = spy_return.gt(0.0)
            upside_deltas.append(
                float(
                    (
                        candidate.returns.reindex(index).fillna(0.0)
                        - base.returns.reindex(index).fillna(0.0)
                    ).loc[upside].mean()
                    * 252.0
                )
            )
            is_focus = strategy == config.primary_strategy
            if is_focus and (
                float(candidate_stats["max_drawdown"]) < HARD_DRAWDOWN_LIMIT
                or drawdown_delta < -FOCUS_DRAWDOWN_DAMAGE_BUDGET
            ):
                feasible = False
        median_cagr = float(np.median(cagr_deltas))
        median_drawdown = float(np.median(drawdown_deltas))
        median_upside = float(np.median(upside_deltas))
        score = (
            median_cagr
            + 0.25 * median_upside
            - 0.75 * max(0.0, -median_drawdown)
        )
        family_hard_tail = float(np.quantile(drawdown_deltas, 0.20))
        score -= 0.50 * max(
            0.0,
            -family_hard_tail - 2.0 * FOCUS_DRAWDOWN_DAMAGE_BUDGET,
        )
        if not feasible:
            score -= 1.0
        if score > best_score:
            best_score = score
            best_parameters = parameters
    if best_parameters is None:
        raise ValueError(f"No policy parameters were evaluated for {policy}.")
    return best_parameters, best_score


def evaluate_policy_results(
    baseline_run: BaselineRun,
    config: BotConfig,
    strategy_families: dict[str, str],
    policy_results: dict[tuple[str, str], BacktestResult],
    oos_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oos_start = pd.Timestamp(oos_predictions["origin_date"].min())
    oos_end = pd.Timestamp(oos_predictions["origin_date"].max())
    strategy_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    era_rows: list[dict[str, object]] = []
    crisis_rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        base = baseline_run.results[strategy]
        for policy in POLICY_ROSTER:
            candidate = policy_results[(strategy, policy)]
            base_stats = _return_slice_stats(base.returns, oos_start, oos_end)
            candidate_stats = _return_slice_stats(candidate.returns, oos_start, oos_end)
            base_cagr = float(base_stats["annualized_return"])
            candidate_cagr = float(candidate_stats["annualized_return"])
            base_dd = float(base_stats["max_drawdown"])
            candidate_dd = float(candidate_stats["max_drawdown"])
            base_wealth = float(
                terminal_wealth_from_cagr(
                    base_cagr,
                    years=15,
                    starting_account_value=220_000.0,
                    annual_contribution=4_000.0,
                    contribution_timing="monthly",
                ).iloc[0]
            )
            candidate_wealth = float(
                terminal_wealth_from_cagr(
                    candidate_cagr,
                    years=15,
                    starting_account_value=220_000.0,
                    annual_contribution=4_000.0,
                    contribution_timing="monthly",
                ).iloc[0]
            )
            base_defense = effective_defensive_weight_path(base.weights)
            candidate_defense = effective_defensive_weight_path(candidate.weights)
            difference = (base_defense - candidate_defense).abs()
            oos_index = base.returns.loc[oos_start:oos_end].index
            spy_returns = (
                daily_returns(baseline_run.prices)["SPY"]
                .reindex(oos_index)
                .fillna(0.0)
            )
            return_delta = (
                candidate.returns.reindex(oos_index).fillna(0.0)
                - base.returns.reindex(oos_index).fillna(0.0)
            )
            up_market = spy_returns.gt(0.0)
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "family": family,
                    "policy": policy,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "base_cagr": base_cagr,
                    "candidate_cagr": candidate_cagr,
                    "cagr_delta": candidate_cagr - base_cagr,
                    "base_max_drawdown": base_dd,
                    "candidate_max_drawdown": candidate_dd,
                    "max_drawdown_delta": candidate_dd - base_dd,
                    "base_calmar": base_cagr / abs(base_dd) if base_dd else np.nan,
                    "candidate_calmar": (
                        candidate_cagr / abs(candidate_dd) if candidate_dd else np.nan
                    ),
                    "wealth_delta_15y": candidate_wealth - base_wealth,
                    "active_day_rate": float(difference.loc[oos_start:oos_end].gt(0.025).mean()),
                    "mean_absolute_defense_change": float(
                        difference.loc[oos_start:oos_end].mean()
                    ),
                    "average_turnover_delta": float(
                        candidate.turnover.loc[oos_start:oos_end].mean()
                        - base.turnover.loc[oos_start:oos_end].mean()
                    ),
                    "transaction_cost_delta": float(
                        candidate.transaction_costs.loc[oos_start:oos_end].sum()
                        - base.transaction_costs.loc[oos_start:oos_end].sum()
                    ),
                    "up_market_annualized_return_delta": float(
                        return_delta.loc[up_market].mean() * 252.0
                    ),
                    "down_market_annualized_return_delta": float(
                        return_delta.loc[~up_market].mean() * 252.0
                    ),
                    "up_market_daily_regret_p10": float(
                        return_delta.loc[up_market].quantile(0.10)
                    ),
                    "allocation_authority": 0.0,
                }
            )
            for fold_name, start_text, end_text in OUTER_FOLDS:
                base_fold = _return_slice_stats(base.returns, start_text, end_text)
                candidate_fold = _return_slice_stats(
                    candidate.returns,
                    start_text,
                    end_text,
                )
                if int(base_fold["observations"]) == 0:
                    continue
                fold_rows.append(
                    {
                        "strategy": strategy,
                        "family": family,
                        "policy": policy,
                        "outer_fold": fold_name,
                        "annualized_return_delta": float(
                            candidate_fold["annualized_return"]
                        )
                        - float(base_fold["annualized_return"]),
                        "max_drawdown_delta": float(candidate_fold["max_drawdown"])
                        - float(base_fold["max_drawdown"]),
                    }
                )
            for era, start_text, end_text in ERA_WINDOWS:
                if pd.Timestamp(end_text) < oos_start:
                    continue
                base_era = _return_slice_stats(
                    base.returns,
                    max(pd.Timestamp(start_text), oos_start),
                    min(pd.Timestamp(end_text), oos_end),
                )
                candidate_era = _return_slice_stats(
                    candidate.returns,
                    max(pd.Timestamp(start_text), oos_start),
                    min(pd.Timestamp(end_text), oos_end),
                )
                if int(base_era["observations"]) == 0:
                    continue
                era_rows.append(
                    {
                        "strategy": strategy,
                        "family": family,
                        "policy": policy,
                        "era": era,
                        "annualized_return_delta": float(
                            candidate_era["annualized_return"]
                        )
                        - float(base_era["annualized_return"]),
                        "max_drawdown_delta": float(candidate_era["max_drawdown"])
                        - float(base_era["max_drawdown"]),
                    }
                )
            for crisis, start_text, end_text in CRISIS_WINDOWS:
                if pd.Timestamp(end_text) < oos_start:
                    continue
                base_crisis = _return_slice_stats(base.returns, start_text, end_text)
                candidate_crisis = _return_slice_stats(
                    candidate.returns,
                    start_text,
                    end_text,
                )
                if int(base_crisis["observations"]) == 0:
                    continue
                crisis_rows.append(
                    {
                        "strategy": strategy,
                        "family": family,
                        "policy": policy,
                        "crisis": crisis,
                        "return_delta": float(candidate_crisis["cumulative_return"])
                        - float(base_crisis["cumulative_return"]),
                        "max_drawdown_delta": float(candidate_crisis["max_drawdown"])
                        - float(base_crisis["max_drawdown"]),
                    }
                )
    strategy_metrics = pd.DataFrame(strategy_rows)
    cost_sensitivity = build_policy_cost_sensitivity(
        baseline_run,
        config,
        policy_results,
        oos_start=oos_start,
        oos_end=oos_end,
    )
    return (
        strategy_metrics,
        pd.DataFrame(fold_rows),
        pd.DataFrame(era_rows),
        pd.DataFrame(crisis_rows),
        cost_sensitivity,
    )


def build_policy_cost_sensitivity(
    baseline_run: BaselineRun,
    config: BotConfig,
    policy_results: dict[tuple[str, str], BacktestResult],
    *,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> pd.DataFrame:
    strategy = config.primary_strategy
    base = baseline_run.results[strategy]
    rows: list[dict[str, object]] = []
    for policy in POLICY_ROSTER:
        weights = policy_results[(strategy, policy)].weights
        for cost_bps in (5.0, 10.0, 20.0):
            base_cost = _result_from_execution_weights(
                base,
                baseline_run.prices,
                base.weights,
                transaction_cost_bps=cost_bps,
                name=f"{strategy}__base__{cost_bps:g}",
            )
            candidate_cost = _result_from_execution_weights(
                base,
                baseline_run.prices,
                weights,
                transaction_cost_bps=cost_bps,
                name=f"{strategy}__{policy}__{cost_bps:g}",
            )
            base_stats = _return_slice_stats(base_cost.returns, oos_start, oos_end)
            candidate_stats = _return_slice_stats(
                candidate_cost.returns,
                oos_start,
                oos_end,
            )
            rows.append(
                {
                    "policy": policy,
                    "cost_bps": cost_bps,
                    "cagr_delta": float(candidate_stats["annualized_return"])
                    - float(base_stats["annualized_return"]),
                    "max_drawdown_delta": float(candidate_stats["max_drawdown"])
                    - float(base_stats["max_drawdown"]),
                }
            )
    return pd.DataFrame(rows)


def build_execution_sensitivity(
    baseline_run: BaselineRun,
    config: BotConfig,
    policy_results: dict[tuple[str, str], BacktestResult],
    *,
    oos_predictions: pd.DataFrame,
) -> pd.DataFrame:
    strategy = config.primary_strategy
    base = baseline_run.results[strategy]
    oos_start = pd.Timestamp(oos_predictions["origin_date"].min())
    oos_end = pd.Timestamp(oos_predictions["origin_date"].max())
    base_stats = _return_slice_stats(base.returns, oos_start, oos_end)
    rows: list[dict[str, object]] = []
    for policy in POLICY_ROSTER:
        candidate_weights = policy_results[(strategy, policy)].weights
        for extra_lag in (0, 1, 2):
            if extra_lag:
                delayed = candidate_weights.shift(extra_lag)
                delayed.iloc[:extra_lag] = (
                    base.weights.reindex(delayed.index)
                    .reindex(columns=delayed.columns, fill_value=0.0)
                    .iloc[:extra_lag]
                )
            else:
                delayed = candidate_weights
            candidate = _result_from_execution_weights(
                base,
                baseline_run.prices,
                delayed,
                transaction_cost_bps=float(
                    config.execution.transaction_cost_bps
                ),
                name=f"{strategy}__{policy}__lag{extra_lag}",
            )
            stats = _return_slice_stats(candidate.returns, oos_start, oos_end)
            rows.append(
                {
                    "policy": policy,
                    "extra_execution_lag_sessions": extra_lag,
                    "cagr_delta": float(stats["annualized_return"])
                    - float(base_stats["annualized_return"]),
                    "max_drawdown_delta": float(stats["max_drawdown"])
                    - float(base_stats["max_drawdown"]),
                }
            )
    return pd.DataFrame(rows)


def build_paired_block_bootstrap(
    baseline_run: BaselineRun,
    config: BotConfig,
    policy_results: dict[tuple[str, str], BacktestResult],
    *,
    oos_predictions: pd.DataFrame,
    block_sessions: int = 63,
    simulations: int = 1_000,
    seed: int = 2_207,
) -> pd.DataFrame:
    strategy = config.primary_strategy
    base = baseline_run.results[strategy]
    oos_start = pd.Timestamp(oos_predictions["origin_date"].min())
    oos_end = pd.Timestamp(oos_predictions["origin_date"].max())
    base_returns = base.returns.loc[oos_start:oos_end].fillna(0.0).to_numpy()
    rows: list[dict[str, object]] = []
    generator = np.random.default_rng(seed)
    for policy in POLICY_ROSTER:
        candidate_returns = (
            policy_results[(strategy, policy)]
            .returns.loc[oos_start:oos_end]
            .fillna(0.0)
            .to_numpy()
        )
        length = min(len(base_returns), len(candidate_returns))
        paired = np.column_stack(
            [base_returns[:length], candidate_returns[:length]]
        )
        starts = np.arange(max(1, length - block_sessions + 1))
        cagr_delta: list[float] = []
        drawdown_delta: list[float] = []
        blocks_needed = int(np.ceil(length / block_sessions))
        for _simulation in range(simulations):
            chosen = generator.choice(starts, size=blocks_needed, replace=True)
            sampled = np.concatenate(
                [
                    paired[start : start + block_sessions]
                    for start in chosen
                ],
                axis=0,
            )[:length]
            base_equity = np.cumprod(1.0 + sampled[:, 0])
            candidate_equity = np.cumprod(1.0 + sampled[:, 1])
            base_cagr = base_equity[-1] ** (252.0 / length) - 1.0
            candidate_cagr = candidate_equity[-1] ** (252.0 / length) - 1.0
            base_dd = np.min(base_equity / np.maximum.accumulate(base_equity) - 1.0)
            candidate_dd = np.min(
                candidate_equity / np.maximum.accumulate(candidate_equity) - 1.0
            )
            cagr_delta.append(float(candidate_cagr - base_cagr))
            drawdown_delta.append(float(candidate_dd - base_dd))
        cagr_array = np.asarray(cagr_delta)
        drawdown_array = np.asarray(drawdown_delta)
        rows.append(
            {
                "policy": policy,
                "block_sessions": block_sessions,
                "simulations": simulations,
                "cagr_delta_p05": float(np.quantile(cagr_array, 0.05)),
                "cagr_delta_p50": float(np.quantile(cagr_array, 0.50)),
                "cagr_delta_p95": float(np.quantile(cagr_array, 0.95)),
                "probability_cagr_delta_positive": float(
                    np.mean(cagr_array > 0.0)
                ),
                "max_drawdown_delta_p05": float(
                    np.quantile(drawdown_array, 0.05)
                ),
                "max_drawdown_delta_p50": float(
                    np.quantile(drawdown_array, 0.50)
                ),
                "max_drawdown_delta_p95": float(
                    np.quantile(drawdown_array, 0.95)
                ),
                "probability_drawdown_damage_over_1pp": float(
                    np.mean(drawdown_array < -FOCUS_DRAWDOWN_DAMAGE_BUDGET)
                ),
            }
        )
    return pd.DataFrame(rows)


def _modal_policy_parameters(
    selections: pd.DataFrame,
    *,
    family: str,
    policy: str,
) -> dict[str, float]:
    selected = selections[
        selections["family"].eq(family)
        & selections["policy"].eq(policy)
    ]
    columns = [
        "defense_floor",
        "hazard_slope",
        "defense_ceiling",
        "native_blend",
        "break_acceleration",
        "age_decay",
    ]
    return {
        column: float(selected[column].mode().iloc[0])
        for column in columns
    }


def build_current_policy_read(
    baseline_run: BaselineRun,
    strategy_families: dict[str, str],
    current_predictions: pd.DataFrame,
    policy_selection: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, family in strategy_families.items():
        base = baseline_run.results[strategy]
        base_weights = base.weights.tail(1)
        base_defense = effective_defensive_weight_path(base_weights)
        prediction = current_predictions[
            current_predictions["family"].eq(family)
        ].iloc[0]
        for policy in POLICY_ROSTER:
            parameters = _modal_policy_parameters(
                policy_selection,
                family=family,
                policy=policy,
            )
            probability_column = _policy_probability_column(policy)
            probability = pd.Series(
                [float(prediction[probability_column])],
                index=base_weights.index,
            )
            target = continuous_defense_target(
                probability,
                base_defense,
                pd.Series(
                    [float(prediction["break_count"])],
                    index=base_weights.index,
                ),
                pd.Series(
                    [float(prediction["non_deterioration_age_weeks"])],
                    index=base_weights.index,
                ),
                policy=policy,
                parameters=parameters,
            )
            adjusted = apply_continuous_defense_budget(
                base_weights,
                target,
                spy_bridge=policy.endswith("spy_bridge"),
            )
            adjusted_defense = effective_defensive_weight_path(adjusted)
            rows.append(
                {
                    "market_date": base_weights.index[-1],
                    "strategy": strategy,
                    "family": family,
                    "policy": policy,
                    "hazard_probability": float(probability.iloc[0]),
                    "base_defensive_weight": float(base_defense.iloc[0]),
                    "target_defensive_weight": float(target.iloc[0]),
                    "adjusted_defensive_weight": float(adjusted_defense.iloc[0]),
                    "defensive_weight_change": float(
                        adjusted_defense.iloc[0] - base_defense.iloc[0]
                    ),
                    "allocation_authority": 0.0,
                    **parameters,
                }
            )
    return pd.DataFrame(rows)


def build_shadow_candidate(
    config: BotConfig,
    *,
    policy_selection: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    execution_sensitivity: pd.DataFrame,
    block_bootstrap: pd.DataFrame,
    current_read: pd.DataFrame,
) -> pd.DataFrame:
    policy = "constant_mild_continuous_existing"
    family = "i111"
    parameters = _modal_policy_parameters(
        policy_selection,
        family=family,
        policy=policy,
    )
    focus = policy_metrics[
        policy_metrics["strategy"].eq(config.primary_strategy)
        & policy_metrics["policy"].eq(policy)
    ].iloc[0]
    lag = execution_sensitivity[
        execution_sensitivity["policy"].eq(policy)
        & execution_sensitivity["extra_execution_lag_sessions"].eq(1)
    ].iloc[0]
    bootstrap = block_bootstrap[
        block_bootstrap["policy"].eq(policy)
    ].iloc[0]
    current = current_read[
        current_read["strategy"].eq(config.primary_strategy)
        & current_read["policy"].eq(policy)
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                "candidate": "i111_continuous_defense_calibration_v1",
                "status": "prospective_shadow_frozen",
                "family_scope": family,
                "focus_strategy": config.primary_strategy,
                "policy": policy,
                "shadow_start_after_market_date": current["market_date"],
                "decision_frequency": "weekly",
                "extra_execution_lag_sessions": 1,
                "probability_input": "expanding_episode_weighted_base_rate",
                "retrospective_focus_cagr_delta": focus["cagr_delta"],
                "retrospective_focus_max_drawdown_delta": focus[
                    "max_drawdown_delta"
                ],
                "one_lag_focus_cagr_delta": lag["cagr_delta"],
                "one_lag_focus_max_drawdown_delta": lag[
                    "max_drawdown_delta"
                ],
                "bootstrap_cagr_delta_p05": bootstrap["cagr_delta_p05"],
                "bootstrap_probability_cagr_positive": bootstrap[
                    "probability_cagr_delta_positive"
                ],
                "bootstrap_probability_dd_damage_over_1pp": bootstrap[
                    "probability_drawdown_damage_over_1pp"
                ],
                "base_defensive_weight_at_freeze": current[
                    "base_defensive_weight"
                ],
                "shadow_defensive_weight_at_freeze": current[
                    "adjusted_defensive_weight"
                ],
                "allocation_authority": 0.0,
                "automatic_promotion_allowed": False,
                "required_evidence": (
                    "new prospective observations; no same-history retuning"
                ),
                **parameters,
            }
        ]
    )


def freeze_shadow_candidate(
    output_dir: str | Path,
    proposed: pd.DataFrame,
) -> pd.DataFrame:
    path = Path(output_dir) / "shadow_candidate.csv"
    if not path.exists():
        return proposed
    existing = pd.read_csv(path)
    if existing.empty:
        return proposed
    expected_name = str(proposed.iloc[0]["candidate"])
    frozen = existing[
        existing["candidate"].astype(str).eq(expected_name)
        & existing["status"].astype(str).eq("prospective_shadow_frozen")
    ]
    return frozen.head(1).copy() if not frozen.empty else proposed


def build_promotion_gates(
    config: BotConfig,
    model_diagnostics: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    leave_crisis_out_diagnostics: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    crisis_metrics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    core_brier = model_diagnostics[
        model_diagnostics["model"].eq("market_core_global")
    ]["brier_score"].mean()
    augmented_brier = model_diagnostics[
        model_diagnostics["model"].eq("market_augmented_global")
    ]["brier_score"].mean()
    family_brier = model_diagnostics[
        model_diagnostics["model"].eq("family_partial_pool")
    ]["brier_score"].mean()
    base_brier = model_diagnostics["base_rate_brier_score"].mean()
    for policy in POLICY_ROSTER:
        focus = policy_metrics[
            policy_metrics["strategy"].eq(config.primary_strategy)
            & policy_metrics["policy"].eq(policy)
        ].iloc[0]
        i111 = policy_metrics[
            policy_metrics["family"].eq("i111")
            & policy_metrics["policy"].eq(policy)
        ]
        focus_folds = fold_metrics[
            fold_metrics["strategy"].eq(config.primary_strategy)
            & fold_metrics["policy"].eq(policy)
        ]
        crises = crisis_metrics[crisis_metrics["policy"].eq(policy)]
        costs = cost_sensitivity[
            cost_sensitivity["policy"].eq(policy)
            & cost_sensitivity["cost_bps"].isin([10.0, 20.0])
        ]
        model_name = (
            "constant_base_rate"
            if policy.startswith("constant_")
            else (
            "market_augmented_global"
            if policy.startswith("global_")
            else "family_partial_pool"
            )
        )
        if model_name == "constant_base_rate":
            model_brier = base_brier
        else:
            model_brier = (
                augmented_brier
                if model_name.startswith("market_")
                else family_brier
            )
        threshold_model = threshold_sensitivity[
            threshold_sensitivity["model"].eq(model_name)
        ]
        leave_out_model = leave_crisis_out_diagnostics[
            leave_crisis_out_diagnostics["model"].eq(model_name)
        ]
        gates = {
            "hazard_beats_base_rate": model_brier < base_brier,
            "incremental_features_beat_core": min(augmented_brier, family_brier)
            < core_brier,
            "threshold_direction_stable": bool(
                not threshold_model.empty
                and float(
                    threshold_model.groupby("break_drawdown_threshold")[
                        "brier_improvement_vs_base_rate"
                    ]
                    .mean()
                    .gt(0.0)
                    .mean()
                )
                >= 2.0 / 3.0
            ),
            "leave_crisis_out_majority": bool(
                not leave_out_model.empty
                and float(
                    leave_out_model["brier_improvement_vs_base_rate"]
                    .gt(0.0)
                    .mean()
                )
                >= 0.50
            ),
            "focus_cagr_positive": float(focus["cagr_delta"]) > 0.0,
            "focus_drawdown_budget": float(focus["max_drawdown_delta"])
            >= -FOCUS_DRAWDOWN_DAMAGE_BUDGET
            and float(focus["candidate_max_drawdown"]) >= HARD_DRAWDOWN_LIMIT,
            "focus_wealth_positive": float(focus["wealth_delta_15y"]) > 0.0,
            "three_of_four_folds_positive": int(
                focus_folds["annualized_return_delta"].gt(0.0).sum()
            )
            >= 3,
            "i111_family_positive": float(i111["cagr_delta"].median()) > 0.0,
            "crisis_drawdown_budget": float(
                crises["max_drawdown_delta"]
                .ge(-CRISIS_DRAWDOWN_DAMAGE_BUDGET)
                .mean()
            )
            >= 0.75,
            "higher_cost_robust": bool(
                not costs.empty and costs["cagr_delta"].gt(0.0).all()
            ),
            "material_allocation_effect": float(focus["active_day_rate"]) >= 0.05
            and float(focus["mean_absolute_defense_change"]) >= 0.025,
        }
        failed = [gate for gate, passed in gates.items() if not passed]
        posthoc_boundary_extension = "mild_continuous" in policy
        if posthoc_boundary_extension:
            failed.append("posthoc_boundary_extension_requires_new_holdout")
        rows.append(
            {
                "policy": policy,
                "model": model_name,
                "model_brier_score": model_brier,
                "base_rate_brier_score": base_brier,
                "core_model_brier_score": core_brier,
                "focus_cagr_delta": focus["cagr_delta"],
                "focus_max_drawdown_delta": focus["max_drawdown_delta"],
                "focus_wealth_delta_15y": focus["wealth_delta_15y"],
                "focus_active_day_rate": focus["active_day_rate"],
                "focus_mean_absolute_defense_change": focus[
                    "mean_absolute_defense_change"
                ],
                "positive_focus_folds": int(
                    focus_folds["annualized_return_delta"].gt(0.0).sum()
                ),
                "i111_median_cagr_delta": i111["cagr_delta"].median(),
                "crisis_drawdown_budget_rate": crises["max_drawdown_delta"]
                .ge(-CRISIS_DRAWDOWN_DAMAGE_BUDGET)
                .mean(),
                **gates,
                "gate_pass_count": sum(gates.values()),
                "failed_gates": ", ".join(failed),
                "posthoc_boundary_extension": posthoc_boundary_extension,
                "retrospective_gate_passed": (
                    all(gates.values()) and not posthoc_boundary_extension
                ),
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["retrospective_gate_passed", "gate_pass_count", "focus_cagr_delta"],
        ascending=[False, False, False],
    )


def write_native_timing_outputs(
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
    summary = root / "summary.md"
    summary.write_text(
        build_native_timing_summary(
            frames,
            primary_strategy=config.primary_strategy,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["summary"] = summary
    manifest = write_research_manifest(
        root,
        study="native_timing_hazard_research",
        config=config,
        prices=prices,
        parameters={
            "break_horizon_sessions": BREAK_HORIZON_SESSIONS,
            "break_drawdown_threshold": BREAK_DRAWDOWN_THRESHOLD,
            "model_roster": list(MODEL_ROSTER),
            "policy_roster": list(POLICY_ROSTER),
            "outer_folds": list(OUTER_FOLDS),
            "regularization_grid": list(REGULARIZATION_GRID),
            "family_prior_strengths": list(FAMILY_PRIOR_STRENGTHS),
            "focus_drawdown_damage_budget": FOCUS_DRAWDOWN_DAMAGE_BUDGET,
            "hard_drawdown_limit": HARD_DRAWDOWN_LIMIT,
            "automatic_promotion_allowed": False,
            "allocation_authority": 0.0,
            "trial_roster": list(MODEL_ROSTER) + list(POLICY_ROSTER),
        },
        artifacts=[path.name for path in paths.values()],
    )
    paths["manifest"] = manifest
    return paths


def build_native_timing_summary(
    frames: dict[str, pd.DataFrame],
    *,
    primary_strategy: str,
) -> str:
    gates = frames["promotion_gates"]
    diagnostics = frames["model_diagnostics"]
    top = gates.iloc[0]
    current = frames["current_read"]
    focus_current = current[
        current["policy"].eq(top["policy"])
        & current["strategy"].eq(primary_strategy)
    ]
    current_change = (
        float(focus_current["defensive_weight_change"].iloc[0])
        if not focus_current.empty
        else np.nan
    )
    return "\n".join(
        [
            "# Native Timing Hazard Research",
            "",
            "Status: nested retrospective research only; allocation authority is 0%.",
            "",
            f"- Model variants: {len(MODEL_ROSTER)}.",
            f"- Continuous policy architectures: {len(POLICY_ROSTER)}.",
            f"- Outer walk-forward folds: {diagnostics['outer_fold'].nunique()}.",
            f"- Full retrospective passes: {int(gates['retrospective_gate_passed'].sum())}.",
            f"- Closest policy: `{top['policy']}`.",
            f"- Gates passed: {int(top['gate_pass_count'])}/12.",
            f"- Failed gates: {top['failed_gates']}.",
            f"- Focus OOS CAGR delta: {float(top['focus_cagr_delta']):.2%}.",
            f"- Focus OOS max-drawdown delta: {float(top['focus_max_drawdown_delta']):.2%}.",
            f"- Focus 15-year wealth delta: ${float(top['focus_wealth_delta_15y']):,.0f}.",
            f"- Current defensive-weight change: {current_change:.2%}.",
        ]
    )
