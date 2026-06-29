from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_FACTOR_ATTRIBUTION_BETA_DRIFT_THRESHOLD,
    DEFAULT_FACTOR_ATTRIBUTION_FACTOR_SPECS,
    DEFAULT_FACTOR_ATTRIBUTION_MIN_OBSERVATIONS,
    DEFAULT_FACTOR_ATTRIBUTION_R2_DROP_THRESHOLD,
    DEFAULT_FACTOR_ATTRIBUTION_RECENT_LOOKBACK_DAYS,
    DEFAULT_FACTOR_ATTRIBUTION_RESIDUAL_VOL_RATIO_THRESHOLD,
    TRADING_DAYS_PER_YEAR,
)

FACTOR_ATTRIBUTION_COLUMNS = [
    "factor",
    "label",
    "proxy_ticker",
    "description",
    "beta",
    "correlation",
    "annualized_factor_volatility",
    "return_contribution",
    "absolute_contribution_share",
    "risk_contribution_pct",
]

FACTOR_DECAY_COLUMNS = [
    "factor",
    "label",
    "proxy_ticker",
    "full_beta",
    "recent_beta",
    "beta_drift",
    "abs_beta_drift",
    "full_r_squared",
    "recent_r_squared",
    "r_squared_drop",
    "full_residual_volatility",
    "recent_residual_volatility",
    "residual_volatility_ratio",
    "drift_flag",
    "model_decay_flag",
]

IMPLEMENTATION_SHORTFALL_COLUMNS = [
    "scope",
    "observations",
    "ideal_final_equity",
    "actual_final_equity",
    "shortfall_dollars",
    "ideal_cumulative_return",
    "actual_cumulative_return",
    "shortfall_return",
    "tracking_error",
    "status",
]

TICKET_SHORTFALL_COLUMNS = [
    "ticket_id",
    "decision_id",
    "mode",
    "account",
    "strategy_name",
    "ticker",
    "side",
    "ticket_status",
    "execution_status",
    "reference_price",
    "actual_price",
    "price_slippage_pct",
    "inside_price_band",
    "target_notional",
    "actual_notional",
    "notional_gap",
    "inside_size_band",
    "shortfall_note",
]


@dataclass(frozen=True)
class FactorAttributionRun:
    summary: pd.DataFrame
    factor_attribution: pd.DataFrame
    factor_return_contributions: pd.DataFrame
    residual_returns: pd.Series


def build_factor_attribution(
    strategy_equity: pd.Series,
    prices: pd.DataFrame,
    *,
    factor_specs: tuple[tuple[str, str, str, str], ...] = DEFAULT_FACTOR_ATTRIBUTION_FACTOR_SPECS,
    min_observations: int = DEFAULT_FACTOR_ATTRIBUTION_MIN_OBSERVATIONS,
) -> FactorAttributionRun:
    """Decompose strategy returns into transparent ETF proxy factors.

    This is an explanatory OLS attribution model, not a tradable forecast. The
    residual line is the strategy behavior not explained by the proxy factors.
    """

    strategy_returns = _strategy_returns(strategy_equity)
    factor_returns, metadata = _factor_returns(prices, factor_specs)
    aligned = pd.concat([strategy_returns.rename("strategy_return"), factor_returns], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.shape[0] < min_observations or factor_returns.empty:
        return _empty_attribution_run()

    y = aligned["strategy_return"].astype(float)
    x = aligned.drop(columns=["strategy_return"]).astype(float)
    beta, alpha = _ols_beta_with_intercept(y, x)
    fitted_factor_returns = x.mul(beta, axis=1)
    intercept_returns = pd.Series(alpha, index=y.index, name="intercept")
    residual_returns = (y - fitted_factor_returns.sum(axis=1) - intercept_returns).rename(
        "residual"
    )

    rows = []
    strategy_variance = float(y.var())
    total_abs_contribution = float(
        fitted_factor_returns.sum().abs().sum()
        + abs(float(intercept_returns.sum()))
        + abs(float(residual_returns.sum()))
    )
    for factor in x.columns:
        contribution = fitted_factor_returns[factor]
        factor_meta = metadata[factor]
        rows.append(
            {
                "factor": factor,
                "label": factor_meta["label"],
                "proxy_ticker": factor_meta["proxy_ticker"],
                "description": factor_meta["description"],
                "beta": float(beta[factor]),
                "correlation": float(y.corr(x[factor])),
                "annualized_factor_volatility": float(x[factor].std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
                "return_contribution": float(contribution.sum()),
                "absolute_contribution_share": _safe_share(
                    abs(float(contribution.sum())),
                    total_abs_contribution,
                ),
                "risk_contribution_pct": _safe_share(
                    float(contribution.cov(y)),
                    strategy_variance,
                ),
            }
        )

    residual_contribution = residual_returns + intercept_returns
    residual_sum = float(residual_contribution.sum())
    rows.append(
        {
            "factor": "residual_strategy",
            "label": "Residual strategy behavior",
            "proxy_ticker": "",
            "description": "Return not explained by the proxy factor set.",
            "beta": np.nan,
            "correlation": np.nan,
            "annualized_factor_volatility": float(
                residual_contribution.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            ),
            "return_contribution": residual_sum,
            "absolute_contribution_share": _safe_share(abs(residual_sum), total_abs_contribution),
            "risk_contribution_pct": _safe_share(float(residual_contribution.cov(y)), strategy_variance),
        }
    )
    attribution = pd.DataFrame(rows, columns=FACTOR_ATTRIBUTION_COLUMNS).sort_values(
        "absolute_contribution_share",
        ascending=False,
    )
    predicted = fitted_factor_returns.sum(axis=1) + intercept_returns
    summary = _attribution_summary(y, predicted, residual_contribution, attribution)
    contributions = fitted_factor_returns.copy()
    contributions["intercept"] = intercept_returns
    contributions["residual"] = residual_returns
    contributions.index.name = "date"
    return FactorAttributionRun(
        summary=summary,
        factor_attribution=attribution,
        factor_return_contributions=contributions,
        residual_returns=residual_contribution.rename("residual_strategy_return"),
    )


def build_factor_decay_monitor(
    strategy_equity: pd.Series,
    prices: pd.DataFrame,
    *,
    recent_lookback_days: int = DEFAULT_FACTOR_ATTRIBUTION_RECENT_LOOKBACK_DAYS,
    min_observations: int = DEFAULT_FACTOR_ATTRIBUTION_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    full = build_factor_attribution(
        strategy_equity,
        prices,
        min_observations=min_observations,
    )
    if full.factor_attribution.empty:
        return pd.DataFrame(columns=FACTOR_DECAY_COLUMNS)
    recent_equity = strategy_equity.dropna().tail(recent_lookback_days + 1)
    recent_prices = prices.reindex(strategy_equity.dropna().index).ffill().tail(
        recent_lookback_days + 1
    )
    recent = build_factor_attribution(
        recent_equity,
        recent_prices,
        min_observations=min(20, max(5, recent_lookback_days // 3)),
    )
    if recent.factor_attribution.empty:
        return pd.DataFrame(columns=FACTOR_DECAY_COLUMNS)
    full_betas = _factor_beta_frame(full.factor_attribution, "full_beta")
    recent_betas = _factor_beta_frame(recent.factor_attribution, "recent_beta")
    decay = full_betas.merge(recent_betas, on=["factor", "label", "proxy_ticker"], how="outer")
    decay["beta_drift"] = decay["recent_beta"] - decay["full_beta"]
    decay["abs_beta_drift"] = decay["beta_drift"].abs()
    full_summary = full.summary.iloc[0]
    recent_summary = recent.summary.iloc[0]
    full_r_squared = float(full_summary.get("factor_model_r_squared", np.nan))
    recent_r_squared = float(recent_summary.get("factor_model_r_squared", np.nan))
    full_residual_vol = float(full_summary.get("residual_annualized_volatility", np.nan))
    recent_residual_vol = float(recent_summary.get("residual_annualized_volatility", np.nan))
    decay["full_r_squared"] = full_r_squared
    decay["recent_r_squared"] = recent_r_squared
    decay["r_squared_drop"] = full_r_squared - recent_r_squared
    decay["full_residual_volatility"] = full_residual_vol
    decay["recent_residual_volatility"] = recent_residual_vol
    decay["residual_volatility_ratio"] = _safe_share(recent_residual_vol, full_residual_vol)
    decay["drift_flag"] = decay["abs_beta_drift"] >= DEFAULT_FACTOR_ATTRIBUTION_BETA_DRIFT_THRESHOLD
    decay["model_decay_flag"] = (
        (decay["r_squared_drop"] >= DEFAULT_FACTOR_ATTRIBUTION_R2_DROP_THRESHOLD)
        | (
            decay["residual_volatility_ratio"]
            >= DEFAULT_FACTOR_ATTRIBUTION_RESIDUAL_VOL_RATIO_THRESHOLD
        )
    )
    return decay[FACTOR_DECAY_COLUMNS].sort_values("abs_beta_drift", ascending=False)


def build_implementation_shortfall(
    ideal_equity: pd.Series,
    actual_valuations: pd.DataFrame,
    *,
    actual_equity_column: str = "equity",
    date_column: str = "valuation_date",
    scope: str = "paper_window",
) -> pd.DataFrame:
    if ideal_equity.empty or actual_valuations.empty:
        return pd.DataFrame(columns=IMPLEMENTATION_SHORTFALL_COLUMNS)
    if actual_equity_column not in actual_valuations or date_column not in actual_valuations:
        return pd.DataFrame(columns=IMPLEMENTATION_SHORTFALL_COLUMNS)

    actual = actual_valuations.copy()
    actual[date_column] = pd.to_datetime(actual[date_column], errors="coerce")
    actual = actual.dropna(subset=[date_column]).sort_values(date_column)
    if actual.empty:
        return pd.DataFrame(columns=IMPLEMENTATION_SHORTFALL_COLUMNS)
    actual_series = pd.Series(
        pd.to_numeric(actual[actual_equity_column], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(actual[date_column]),
        name="actual_equity",
    ).dropna()
    ideal = ideal_equity.copy()
    ideal.index = pd.to_datetime(ideal.index)
    aligned_ideal = ideal.reindex(actual_series.index).ffill().dropna()
    actual_series = actual_series.reindex(aligned_ideal.index).dropna()
    aligned_ideal = aligned_ideal.reindex(actual_series.index)
    if len(aligned_ideal) < 2:
        return pd.DataFrame(columns=IMPLEMENTATION_SHORTFALL_COLUMNS)

    rebased_ideal = aligned_ideal / float(aligned_ideal.iloc[0]) * float(actual_series.iloc[0])
    ideal_return = float(rebased_ideal.iloc[-1] / rebased_ideal.iloc[0] - 1.0)
    actual_return = float(actual_series.iloc[-1] / actual_series.iloc[0] - 1.0)
    tracking = actual_series.pct_change().sub(rebased_ideal.pct_change()).dropna()
    shortfall_return = actual_return - ideal_return
    return pd.DataFrame(
        [
            {
                "scope": scope,
                "observations": int(len(actual_series)),
                "ideal_final_equity": float(rebased_ideal.iloc[-1]),
                "actual_final_equity": float(actual_series.iloc[-1]),
                "shortfall_dollars": float(actual_series.iloc[-1] - rebased_ideal.iloc[-1]),
                "ideal_cumulative_return": ideal_return,
                "actual_cumulative_return": actual_return,
                "shortfall_return": shortfall_return,
                "tracking_error": float(tracking.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
                if len(tracking) > 1
                else 0.0,
                "status": _shortfall_status(shortfall_return),
            }
        ],
        columns=IMPLEMENTATION_SHORTFALL_COLUMNS,
    )


def build_ticket_shortfall_audit(
    tickets: pd.DataFrame,
    executions: pd.DataFrame,
) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame(columns=TICKET_SHORTFALL_COLUMNS)
    frame = tickets.copy()
    execution_frame = executions.copy() if not executions.empty else pd.DataFrame()
    if not execution_frame.empty and "recommendation_id" in execution_frame:
        execution_frame = (
            execution_frame.sort_values(["recommendation_id", "executed_at_utc"])
            .drop_duplicates("recommendation_id", keep="last")
            .rename(
                columns={
                    "recommendation_id": "ticket_id",
                    "price": "actual_price",
                    "notional": "actual_notional",
                }
            )
        )
        frame = frame.merge(
            execution_frame[["ticket_id", "actual_price", "actual_notional"]],
            on="ticket_id",
            how="left",
        )
    else:
        frame["actual_price"] = np.nan
        frame["actual_notional"] = np.nan

    frame["execution_status"] = np.where(frame["actual_price"].notna(), "executed", "not_executed")
    frame["ticket_status"] = frame.get("status", "").astype(str)
    frame["price_slippage_pct"] = (
        pd.to_numeric(frame["actual_price"], errors="coerce")
        / pd.to_numeric(frame["reference_price"], errors="coerce")
        - 1.0
    )
    frame["inside_price_band"] = (
        (pd.to_numeric(frame["actual_price"], errors="coerce") >= frame["limit_low"].astype(float))
        & (pd.to_numeric(frame["actual_price"], errors="coerce") <= frame["limit_high"].astype(float))
    )
    frame.loc[frame["actual_price"].isna(), "inside_price_band"] = False
    frame["notional_gap"] = (
        pd.to_numeric(frame["actual_notional"], errors="coerce").fillna(0.0)
        - frame["target_notional"].astype(float).abs()
    )
    frame["inside_size_band"] = (
        (pd.to_numeric(frame["actual_notional"], errors="coerce") >= frame["min_notional"].astype(float))
        & (pd.to_numeric(frame["actual_notional"], errors="coerce") <= frame["max_notional"].astype(float))
    )
    frame.loc[frame["actual_notional"].isna(), "inside_size_band"] = False
    frame["shortfall_note"] = frame.apply(_ticket_shortfall_note, axis=1)
    return frame[
        [column for column in TICKET_SHORTFALL_COLUMNS if column in frame.columns]
    ].sort_values(["execution_status", "ticker"])


def _strategy_returns(strategy_equity: pd.Series) -> pd.Series:
    equity = strategy_equity.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    returns.index = pd.to_datetime(returns.index)
    return returns.rename("strategy_return")


def _factor_returns(
    prices: pd.DataFrame,
    factor_specs: tuple[tuple[str, str, str, str], ...],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    returns = prices.sort_index().astype(float).pct_change().replace([np.inf, -np.inf], np.nan)
    frame = pd.DataFrame(index=returns.index)
    metadata: dict[str, dict[str, str]] = {}
    for factor, proxy_ticker, label, description in factor_specs:
        if proxy_ticker not in returns:
            continue
        frame[factor] = returns[proxy_ticker]
        metadata[factor] = {
            "proxy_ticker": proxy_ticker,
            "label": label,
            "description": description,
        }
    frame.index = pd.to_datetime(frame.index)
    return frame, metadata


def _ols_beta_with_intercept(y: pd.Series, x: pd.DataFrame) -> tuple[pd.Series, float]:
    design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
    alpha = float(coefficients[0])
    beta = pd.Series(coefficients[1:], index=x.columns, dtype=float)
    return beta, alpha


def _attribution_summary(
    strategy_returns: pd.Series,
    predicted_returns: pd.Series,
    residual_returns: pd.Series,
    attribution: pd.DataFrame,
) -> pd.DataFrame:
    residual_variance = float(residual_returns.var())
    strategy_variance = float(strategy_returns.var())
    residual_vol = float(residual_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    total_return = float(strategy_returns.sum())
    factor_rows = attribution[attribution["factor"] != "residual_strategy"]
    dominant = (
        factor_rows.sort_values("absolute_contribution_share", ascending=False).iloc[0]
        if not factor_rows.empty
        else pd.Series(dtype=object)
    )
    r_squared = 1.0 - _safe_share(float((strategy_returns - predicted_returns).var()), strategy_variance)
    residual_share = _safe_share(abs(float(residual_returns.sum())), attribution["return_contribution"].abs().sum())
    return pd.DataFrame(
        [
            {
                "observations": int(len(strategy_returns)),
                "strategy_arithmetic_return": total_return,
                "factor_model_r_squared": float(np.clip(r_squared, 0.0, 1.0)),
                "residual_return_contribution": float(residual_returns.sum()),
                "residual_contribution_share": residual_share,
                "residual_annualized_volatility": residual_vol,
                "residual_variance_share": _safe_share(residual_variance, strategy_variance),
                "dominant_factor": str(dominant.get("label", "")),
                "dominant_factor_share": float(dominant.get("absolute_contribution_share", np.nan)),
            }
        ]
    )


def _factor_beta_frame(attribution: pd.DataFrame, beta_column: str) -> pd.DataFrame:
    frame = attribution[attribution["factor"] != "residual_strategy"][
        ["factor", "label", "proxy_ticker", "beta"]
    ].copy()
    return frame.rename(columns={"beta": beta_column})


def _empty_attribution_run() -> FactorAttributionRun:
    return FactorAttributionRun(
        summary=pd.DataFrame(),
        factor_attribution=pd.DataFrame(columns=FACTOR_ATTRIBUTION_COLUMNS),
        factor_return_contributions=pd.DataFrame(),
        residual_returns=pd.Series(dtype=float, name="residual_strategy_return"),
    )


def _safe_share(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _shortfall_status(shortfall_return: float) -> str:
    if shortfall_return >= -0.0025:
        return "in_line"
    if shortfall_return >= -0.01:
        return "minor_shortfall"
    return "material_shortfall"


def _ticket_shortfall_note(row: pd.Series) -> str:
    if str(row.get("execution_status", "")) != "executed":
        status = str(row.get("ticket_status", "open"))
        return f"Ticket is {status}; no matching execution is logged."
    if not bool(row.get("inside_price_band", False)):
        return "Executed outside the recommended price band."
    if not bool(row.get("inside_size_band", False)):
        return "Executed outside the recommended size band."
    return "Execution is inside logged price and size bands."
