from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_RISK_AI_BETA_TICKERS,
    DEFAULT_RISK_BASE_MAX_AI_BETA,
    DEFAULT_RISK_BASE_MAX_EQUITY_BETA,
    DEFAULT_RISK_BASE_MAX_EXPECTED_SHORTFALL_95,
    DEFAULT_RISK_BASE_MAX_SCENARIO_WEIGHTED_STRESS_LOSS,
    DEFAULT_RISK_BASE_MAX_STRESS_LOSS,
    DEFAULT_RISK_BASE_MIN_DEFENSIVE_WEIGHT,
    DEFAULT_RISK_BROAD_EQUITY_TICKERS,
    DEFAULT_RISK_COMMODITY_TICKERS,
    DEFAULT_RISK_CORRELATION_LONG_LOOKBACK_DAYS,
    DEFAULT_RISK_CORRELATION_SHIFT_THRESHOLD,
    DEFAULT_RISK_CORRELATION_SHORT_LOOKBACK_DAYS,
    DEFAULT_RISK_COVARIANCE_LOOKBACK_DAYS,
    DEFAULT_RISK_CREDIT_TICKERS,
    DEFAULT_RISK_DEFENSIVE_TICKER,
    DEFAULT_RISK_DEFENSIVE_TICKERS,
    DEFAULT_RISK_DOLLAR_TICKERS,
    DEFAULT_RISK_DURATION_TICKERS,
    DEFAULT_RISK_ENERGY_TICKERS,
    DEFAULT_RISK_EXPECTED_SHORTFALL_LEVELS,
    DEFAULT_RISK_FACTOR_LOOKBACK_DAYS,
    DEFAULT_RISK_FACTOR_PROXIES,
    DEFAULT_RISK_GOLD_TICKERS,
    DEFAULT_RISK_HIGH_BETA_TICKERS,
    DEFAULT_RISK_INTERNATIONAL_TICKERS,
    DEFAULT_RISK_MAX_CONCENTRATION_HHI,
    DEFAULT_RISK_MAX_SINGLE_ASSET_WEIGHT,
    DEFAULT_RISK_MAX_TURNOVER,
    DEFAULT_RISK_MIN_RISK_ASSET_MULTIPLIER,
    DEFAULT_RISK_PRIVATE_CREDIT_TICKERS,
    DEFAULT_RISK_STRESS_TESTS,
    DEFAULT_RISK_TAIL_LOOKBACK_DAYS,
    DEFAULT_RISK_VOLATILITY_TICKERS,
    DEFAULT_SCENARIO_MAX_MULTIPLIER,
    DEFAULT_SCENARIO_MIN_MULTIPLIER,
    RiskStressTestDefinition,
)
from trade_bot.features.indicators import TRADING_DAYS_PER_YEAR, daily_returns


@dataclass(frozen=True)
class PortfolioRiskConfig:
    defensive_ticker: str = DEFAULT_RISK_DEFENSIVE_TICKER
    factor_lookback_days: int = DEFAULT_RISK_FACTOR_LOOKBACK_DAYS
    covariance_lookback_days: int = DEFAULT_RISK_COVARIANCE_LOOKBACK_DAYS
    correlation_short_lookback_days: int = DEFAULT_RISK_CORRELATION_SHORT_LOOKBACK_DAYS
    correlation_long_lookback_days: int = DEFAULT_RISK_CORRELATION_LONG_LOOKBACK_DAYS
    tail_lookback_days: int = DEFAULT_RISK_TAIL_LOOKBACK_DAYS
    expected_shortfall_levels: tuple[float, ...] = DEFAULT_RISK_EXPECTED_SHORTFALL_LEVELS
    max_single_asset_weight: float = DEFAULT_RISK_MAX_SINGLE_ASSET_WEIGHT
    max_concentration_hhi: float = DEFAULT_RISK_MAX_CONCENTRATION_HHI
    base_max_equity_beta: float = DEFAULT_RISK_BASE_MAX_EQUITY_BETA
    base_max_ai_beta: float = DEFAULT_RISK_BASE_MAX_AI_BETA
    base_max_expected_shortfall_95: float = DEFAULT_RISK_BASE_MAX_EXPECTED_SHORTFALL_95
    base_max_stress_loss: float = DEFAULT_RISK_BASE_MAX_STRESS_LOSS
    base_max_scenario_weighted_stress_loss: float = (
        DEFAULT_RISK_BASE_MAX_SCENARIO_WEIGHTED_STRESS_LOSS
    )
    base_min_defensive_weight: float = DEFAULT_RISK_BASE_MIN_DEFENSIVE_WEIGHT
    max_turnover: float = DEFAULT_RISK_MAX_TURNOVER
    correlation_shift_threshold: float = DEFAULT_RISK_CORRELATION_SHIFT_THRESHOLD
    min_risk_asset_multiplier: float = DEFAULT_RISK_MIN_RISK_ASSET_MULTIPLIER
    factor_proxies: tuple[tuple[str, str, str], ...] = DEFAULT_RISK_FACTOR_PROXIES
    stress_tests: tuple[RiskStressTestDefinition, ...] = DEFAULT_RISK_STRESS_TESTS


@dataclass(frozen=True)
class PortfolioRiskRun:
    summary: pd.DataFrame
    factor_exposures: pd.DataFrame
    beta_decomposition: pd.DataFrame
    correlation_regime: pd.DataFrame
    tail_risk: pd.DataFrame
    stress_tests: pd.DataFrame
    marginal_risk_contribution: pd.DataFrame
    constraint_report: pd.DataFrame
    scenario_risk_budget: pd.DataFrame
    sizing_adjustments: pd.DataFrame
    risk_adjusted_weights: pd.Series


@dataclass(frozen=True)
class _ScenarioBudget:
    risk_off_probability: float
    transition_probability: float
    fragile_upside_probability: float
    risk_on_probability: float
    ai_unwind_probability: float
    credit_stress_probability: float
    inflation_oil_probability: float
    scenario_risk_multiplier: float
    max_single_asset_weight: float
    max_concentration_hhi: float
    max_equity_beta: float
    max_ai_beta: float
    max_expected_shortfall_95: float
    max_stress_loss: float
    max_scenario_weighted_stress_loss: float
    min_defensive_weight: float


@dataclass(frozen=True)
class _MetricSnapshot:
    equity_beta: float
    ai_beta: float
    value_at_risk_95: float
    expected_shortfall_95: float
    max_stress_loss: float
    scenario_weighted_stress_loss: float
    max_single_asset_weight: float
    concentration_hhi: float
    defensive_weight: float
    risk_asset_weight: float
    average_correlation_short: float
    average_correlation_long: float
    correlation_shift: float


def current_positions(weights: pd.DataFrame, top_n: int = 10) -> pd.Series:
    latest = weights.iloc[-1].sort_values(ascending=False)
    return latest[latest > 0].head(top_n)


def next_trade_weights(weights: pd.DataFrame) -> pd.Series:
    if len(weights) < 2:
        return weights.iloc[-1]
    return weights.iloc[-1] - weights.iloc[-2]


def build_portfolio_risk(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    scenario_lattice: pd.DataFrame,
    *,
    current_weights: pd.Series | None = None,
    config: PortfolioRiskConfig | None = None,
) -> PortfolioRiskRun:
    risk_config = config or PortfolioRiskConfig()
    candidate_weights = _normalize_weights(target_weights, risk_config.defensive_ticker)
    starting_weights = _normalize_weights(
        current_weights if current_weights is not None else target_weights,
        risk_config.defensive_ticker,
    )
    if prices.empty or candidate_weights.empty:
        return _empty_portfolio_risk_run(candidate_weights)

    returns = daily_returns(prices.sort_index()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scenario_budget = _build_scenario_budget(scenario_lattice, risk_config)
    pre_snapshot = _metric_snapshot(returns, candidate_weights, scenario_budget, risk_config)
    risk_adjusted_weights, applied_constraints = _apply_constraints(
        returns,
        candidate_weights,
        scenario_budget,
        risk_config,
    )
    post_snapshot = _metric_snapshot(returns, risk_adjusted_weights, scenario_budget, risk_config)

    pre_factor_exposures = _factor_exposures(returns, candidate_weights, risk_config)
    post_factor_exposures = _factor_exposures(returns, risk_adjusted_weights, risk_config)
    stress_tests = _stress_tests_frame(
        candidate_weights,
        risk_adjusted_weights,
        scenario_budget,
        risk_config,
    )
    tail_risk = _tail_risk_frame(returns, candidate_weights, risk_adjusted_weights, risk_config)
    correlation_regime = _correlation_regime_frame(
        returns,
        risk_adjusted_weights,
        post_snapshot,
        risk_config,
    )
    constraint_report = _constraint_report_frame(
        pre_snapshot,
        post_snapshot,
        scenario_budget,
        starting_weights,
        risk_adjusted_weights,
        applied_constraints,
        risk_config,
    )
    sizing_adjustments = _sizing_adjustments_frame(
        starting_weights,
        candidate_weights,
        risk_adjusted_weights,
        applied_constraints,
        risk_config,
    )

    return PortfolioRiskRun(
        summary=_summary_frame(
            pre_snapshot,
            post_snapshot,
            scenario_budget,
            applied_constraints,
            risk_config,
        ),
        factor_exposures=post_factor_exposures,
        beta_decomposition=_beta_decomposition_frame(pre_factor_exposures, post_factor_exposures),
        correlation_regime=correlation_regime,
        tail_risk=tail_risk,
        stress_tests=stress_tests,
        marginal_risk_contribution=_marginal_risk_contribution(
            returns,
            risk_adjusted_weights,
            risk_config,
        ),
        constraint_report=constraint_report,
        scenario_risk_budget=_scenario_budget_frame(scenario_budget),
        sizing_adjustments=sizing_adjustments,
        risk_adjusted_weights=risk_adjusted_weights,
    )


def _empty_portfolio_risk_run(weights: pd.Series) -> PortfolioRiskRun:
    return PortfolioRiskRun(
        summary=pd.DataFrame(),
        factor_exposures=pd.DataFrame(),
        beta_decomposition=pd.DataFrame(),
        correlation_regime=pd.DataFrame(),
        tail_risk=pd.DataFrame(),
        stress_tests=pd.DataFrame(),
        marginal_risk_contribution=pd.DataFrame(),
        constraint_report=pd.DataFrame(),
        scenario_risk_budget=pd.DataFrame(),
        sizing_adjustments=pd.DataFrame(),
        risk_adjusted_weights=weights,
    )


def _normalize_weights(weights: pd.Series, defensive_ticker: str) -> pd.Series:
    if weights.empty:
        return pd.Series(dtype=float)
    clean = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if defensive_ticker not in clean.index:
        clean.loc[defensive_ticker] = 0.0
    total = float(clean.sum())
    if total > 1.0:
        clean = clean / total
    return clean[clean.abs() > 1e-10].sort_values(ascending=False)


def _build_scenario_budget(
    scenario_lattice: pd.DataFrame,
    config: PortfolioRiskConfig,
) -> _ScenarioBudget:
    one_month = _one_month_scenarios(scenario_lattice)
    risk_off = _probability_where(one_month, "risk_bucket", "risk_off")
    transition = _probability_equals(one_month, "risk_bucket", "transition")
    fragile = _probability_equals(one_month, "risk_bucket", "risk_on_fragile")
    risk_on = _probability_equals(one_month, "risk_bucket", "risk_on")
    ai_unwind = _theme_probability(one_month, ("ai", "capex", "semiconductor", "growth"))
    credit_stress = _theme_probability(one_month, ("credit", "spread", "private_credit"))
    inflation_oil = _theme_probability(one_month, ("oil", "inflation", "commodity", "energy"))

    scenario_multiplier = 1.0 - 0.55 * risk_off - 0.20 * transition - 0.15 * fragile
    scenario_multiplier = float(
        np.clip(
            scenario_multiplier, DEFAULT_SCENARIO_MIN_MULTIPLIER, DEFAULT_SCENARIO_MAX_MULTIPLIER
        )
    )
    max_equity_beta = max(
        0.35,
        config.base_max_equity_beta * (1.0 - 0.35 * risk_off - 0.15 * transition),
    )
    max_ai_beta = max(
        0.20,
        config.base_max_ai_beta * (1.0 - 0.45 * ai_unwind - 0.30 * risk_off - 0.15 * fragile),
    )
    max_expected_shortfall = max(
        0.0125,
        config.base_max_expected_shortfall_95 * (1.0 - 0.35 * risk_off - 0.15 * transition),
    )
    max_stress_loss = max(
        0.06,
        config.base_max_stress_loss * (1.0 - 0.35 * risk_off - 0.15 * transition),
    )
    max_scenario_weighted_stress_loss = max(
        0.03,
        config.base_max_scenario_weighted_stress_loss * (1.0 - 0.25 * risk_off - 0.10 * transition),
    )
    min_defensive = float(
        np.clip(
            config.base_min_defensive_weight
            + 0.40 * risk_off
            + 0.20 * transition
            + 0.10 * fragile
            + 0.10 * ai_unwind,
            0.0,
            0.65,
        )
    )
    max_single = max(
        0.25,
        config.max_single_asset_weight * (1.0 - 0.25 * risk_off - 0.10 * transition),
    )
    max_hhi = max(
        0.22,
        config.max_concentration_hhi * (1.0 - 0.25 * risk_off - 0.10 * transition),
    )
    return _ScenarioBudget(
        risk_off_probability=risk_off,
        transition_probability=transition,
        fragile_upside_probability=fragile,
        risk_on_probability=risk_on,
        ai_unwind_probability=ai_unwind,
        credit_stress_probability=credit_stress,
        inflation_oil_probability=inflation_oil,
        scenario_risk_multiplier=scenario_multiplier,
        max_single_asset_weight=max_single,
        max_concentration_hhi=max_hhi,
        max_equity_beta=max_equity_beta,
        max_ai_beta=max_ai_beta,
        max_expected_shortfall_95=max_expected_shortfall,
        max_stress_loss=max_stress_loss,
        max_scenario_weighted_stress_loss=max_scenario_weighted_stress_loss,
        min_defensive_weight=min_defensive,
    )


def _one_month_scenarios(scenario_lattice: pd.DataFrame) -> pd.DataFrame:
    if scenario_lattice.empty:
        return pd.DataFrame()
    if "horizon" not in scenario_lattice:
        return scenario_lattice.copy()
    one_month = scenario_lattice[scenario_lattice["horizon"] == "1m"].copy()
    return one_month if not one_month.empty else scenario_lattice.copy()


def _probability_where(frame: pd.DataFrame, column: str, token: str) -> float:
    if frame.empty or column not in frame or "probability" not in frame:
        return 0.0
    mask = frame[column].astype(str).str.contains(token, case=False, na=False)
    return float(frame.loc[mask, "probability"].astype(float).sum())


def _probability_equals(frame: pd.DataFrame, column: str, value: str) -> float:
    if frame.empty or column not in frame or "probability" not in frame:
        return 0.0
    mask = frame[column].astype(str).str.lower() == value.lower()
    return float(frame.loc[mask, "probability"].astype(float).sum())


def _theme_probability(frame: pd.DataFrame, tokens: tuple[str, ...]) -> float:
    if frame.empty or "probability" not in frame:
        return 0.0
    text_columns = [
        column
        for column in [
            "scenario",
            "scenario_id",
            "family",
            "risk_bucket",
            "expected_bot_posture",
            "preferred_exposure",
            "avoid_exposure",
        ]
        if column in frame
    ]
    if not text_columns:
        return 0.0
    combined = frame[text_columns].astype(str).agg(" ".join, axis=1).str.lower()
    pattern = "|".join(tokens)
    mask = combined.str.contains(pattern, na=False)
    return float(frame.loc[mask, "probability"].astype(float).sum())


def _metric_snapshot(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenario_budget: _ScenarioBudget,
    config: PortfolioRiskConfig,
) -> _MetricSnapshot:
    factor_exposures = _factor_exposures(returns, weights, config)
    factor_beta = _factor_beta_map(factor_exposures)
    tail_stats = _tail_stats(
        _portfolio_returns(returns, weights).tail(config.tail_lookback_days),
        0.95,
    )
    stress_tests = _stress_tests_frame(weights, weights, scenario_budget, config)
    post_loss = stress_tests["post_loss"] if "post_loss" in stress_tests else pd.Series(dtype=float)
    weighted_loss = (
        stress_tests["scenario_probability_weight"] * post_loss
        if "scenario_probability_weight" in stress_tests
        else pd.Series(dtype=float)
    )
    correlation = _correlation_metrics(returns, weights, config)
    return _MetricSnapshot(
        equity_beta=float(factor_beta.get("market_beta", 0.0)),
        ai_beta=max(
            float(factor_beta.get("ai_semiconductor_beta", 0.0)),
            0.75 * float(factor_beta.get("nasdaq_growth_beta", 0.0)),
            1.25 * _group_weight(weights, "ai_beta"),
        ),
        value_at_risk_95=tail_stats["value_at_risk"],
        expected_shortfall_95=tail_stats["expected_shortfall"],
        max_stress_loss=float(post_loss.max()) if not post_loss.empty else 0.0,
        scenario_weighted_stress_loss=(
            float(weighted_loss.sum()) if not weighted_loss.empty else 0.0
        ),
        max_single_asset_weight=float(weights.max()) if not weights.empty else 0.0,
        concentration_hhi=float((weights**2).sum()) if not weights.empty else 0.0,
        defensive_weight=float(weights.get(config.defensive_ticker, 0.0)),
        risk_asset_weight=float(
            weights.drop(labels=[config.defensive_ticker], errors="ignore").sum()
        ),
        average_correlation_short=correlation["short"],
        average_correlation_long=correlation["long"],
        correlation_shift=correlation["shift"],
    )


def _portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if returns.empty or weights.empty:
        return pd.Series(dtype=float)
    aligned_returns = returns.reindex(columns=weights.index).fillna(0.0)
    return (aligned_returns * weights).sum(axis=1).rename("portfolio_return")


def _factor_exposures(
    returns: pd.DataFrame,
    weights: pd.Series,
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    portfolio = _portfolio_returns(returns, weights).tail(config.factor_lookback_days)
    rows: list[dict[str, object]] = []
    for factor, proxy_ticker, description in config.factor_proxies:
        if proxy_ticker not in returns or portfolio.empty:
            rows.append(
                {
                    "factor": factor,
                    "proxy_ticker": proxy_ticker,
                    "proxy_description": description,
                    "beta": np.nan,
                    "correlation": np.nan,
                    "portfolio_annualized_volatility": np.nan,
                    "factor_annualized_volatility": np.nan,
                    "lookback_observations": 0,
                }
            )
            continue
        aligned = pd.concat(
            [portfolio, returns[proxy_ticker].rename("factor_return")],
            axis=1,
        ).dropna()
        aligned = aligned.tail(config.factor_lookback_days)
        if aligned.shape[0] < 20:
            beta = np.nan
            correlation = np.nan
        else:
            factor_var = float(aligned["factor_return"].var())
            beta = (
                float(aligned["portfolio_return"].cov(aligned["factor_return"]) / factor_var)
                if factor_var > 0
                else np.nan
            )
            correlation = float(aligned["portfolio_return"].corr(aligned["factor_return"]))
        rows.append(
            {
                "factor": factor,
                "proxy_ticker": proxy_ticker,
                "proxy_description": description,
                "beta": beta,
                "correlation": correlation,
                "portfolio_annualized_volatility": (
                    float(aligned["portfolio_return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
                    if not aligned.empty
                    else np.nan
                ),
                "factor_annualized_volatility": (
                    float(aligned["factor_return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
                    if not aligned.empty
                    else np.nan
                ),
                "lookback_observations": int(aligned.shape[0]),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["absolute_beta"] = frame["beta"].abs()
        frame = frame.sort_values("absolute_beta", ascending=False, na_position="last")
    return frame


def _factor_beta_map(factor_exposures: pd.DataFrame) -> dict[str, float]:
    if factor_exposures.empty or "factor" not in factor_exposures:
        return {}
    clean = factor_exposures.dropna(subset=["beta"])
    return dict(zip(clean["factor"], clean["beta"], strict=False))


def _tail_stats(portfolio_returns: pd.Series, level: float) -> dict[str, float]:
    clean = portfolio_returns.dropna()
    if clean.empty:
        return {
            "value_at_risk": 0.0,
            "expected_shortfall": 0.0,
            "worst_day": 0.0,
            "realized_volatility": 0.0,
            "observations": 0.0,
        }
    quantile = float(clean.quantile(1.0 - level))
    tail = clean[clean <= quantile]
    expected_shortfall = float(tail.mean()) if not tail.empty else quantile
    return {
        "value_at_risk": abs(min(0.0, quantile)),
        "expected_shortfall": abs(min(0.0, expected_shortfall)),
        "worst_day": float(clean.min()),
        "realized_volatility": float(clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "observations": float(clean.shape[0]),
    }


def _tail_risk_frame(
    returns: pd.DataFrame,
    pre_weights: pd.Series,
    post_weights: pd.Series,
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, weights in [
        ("pre_risk_target", pre_weights),
        ("risk_adjusted", post_weights),
    ]:
        portfolio = _portfolio_returns(returns, weights).tail(config.tail_lookback_days)
        for level in config.expected_shortfall_levels:
            stats = _tail_stats(portfolio, level)
            rows.append(
                {
                    "stage": stage,
                    "confidence_level": level,
                    "value_at_risk": stats["value_at_risk"],
                    "expected_shortfall": stats["expected_shortfall"],
                    "worst_day": stats["worst_day"],
                    "realized_volatility": stats["realized_volatility"],
                    "observations": int(stats["observations"]),
                }
            )
    return pd.DataFrame(rows)


def _stress_tests_frame(
    pre_weights: pd.Series,
    post_weights: pd.Series,
    scenario_budget: _ScenarioBudget,
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stress in config.stress_tests:
        pre_return = _stress_return(pre_weights, stress)
        post_return = _stress_return(post_weights, stress)
        scenario_probability_weight = _stress_probability_weight(stress.name, scenario_budget)
        rows.append(
            {
                "stress": stress.name,
                "description": stress.description,
                "scenario_probability_weight": scenario_probability_weight,
                "pre_shock_return": pre_return,
                "pre_loss": abs(min(0.0, pre_return)),
                "post_shock_return": post_return,
                "post_loss": abs(min(0.0, post_return)),
                "risk_engine_delta_loss": abs(min(0.0, post_return)) - abs(min(0.0, pre_return)),
            }
        )
    return pd.DataFrame(rows).sort_values("post_loss", ascending=False)


def _stress_return(weights: pd.Series, stress: RiskStressTestDefinition) -> float:
    shocks = dict(stress.group_shocks)
    total = 0.0
    for ticker, weight in weights.items():
        group = _ticker_group(str(ticker))
        shock = shocks.get(group, stress.default_shock)
        total += float(weight) * float(shock)
    return float(total)


def _ticker_group(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in DEFAULT_RISK_DEFENSIVE_TICKERS:
        return "defensive"
    if ticker in DEFAULT_RISK_VOLATILITY_TICKERS:
        return "volatility"
    if ticker in DEFAULT_RISK_AI_BETA_TICKERS:
        return "ai_beta"
    if ticker in DEFAULT_RISK_HIGH_BETA_TICKERS:
        return "high_beta"
    if ticker in DEFAULT_RISK_PRIVATE_CREDIT_TICKERS:
        return "private_credit"
    if ticker in DEFAULT_RISK_CREDIT_TICKERS:
        return "credit"
    if ticker in DEFAULT_RISK_DURATION_TICKERS:
        return "duration"
    if ticker in DEFAULT_RISK_ENERGY_TICKERS:
        return "energy"
    if ticker in DEFAULT_RISK_GOLD_TICKERS:
        return "gold"
    if ticker in DEFAULT_RISK_COMMODITY_TICKERS:
        return "commodity"
    if ticker in DEFAULT_RISK_DOLLAR_TICKERS:
        return "dollar"
    if ticker in DEFAULT_RISK_INTERNATIONAL_TICKERS:
        return "international"
    if ticker in DEFAULT_RISK_BROAD_EQUITY_TICKERS:
        return "broad_equity"
    return "other"


def _stress_probability_weight(stress_name: str, scenario_budget: _ScenarioBudget) -> float:
    mapping = {
        "equity_crash": 0.50 * scenario_budget.risk_off_probability,
        "rates_up_shock": (
            0.35 * scenario_budget.transition_probability
            + 0.30 * scenario_budget.inflation_oil_probability
        ),
        "credit_event": (
            0.45 * scenario_budget.risk_off_probability
            + 0.40 * scenario_budget.credit_stress_probability
        ),
        "ai_capex_unwind": (
            0.50 * scenario_budget.ai_unwind_probability
            + 0.35 * scenario_budget.fragile_upside_probability
        ),
        "oil_geopolitical_shock": (
            0.45 * scenario_budget.inflation_oil_probability
            + 0.20 * scenario_budget.transition_probability
        ),
        "dollar_liquidity_squeeze": (
            0.25 * scenario_budget.risk_off_probability
            + 0.25 * scenario_budget.transition_probability
        ),
        "risk_on_relief": scenario_budget.risk_on_probability,
    }
    return float(mapping.get(stress_name, 0.0))


def _correlation_metrics(
    returns: pd.DataFrame,
    weights: pd.Series,
    config: PortfolioRiskConfig,
) -> dict[str, float]:
    tickers = [ticker for ticker, weight in weights.items() if weight > 0.005 and ticker in returns]
    if len(tickers) < 2:
        return {"short": 0.0, "long": 0.0, "shift": 0.0}
    short = _average_pairwise_correlation(
        returns[tickers].tail(config.correlation_short_lookback_days)
    )
    long = _average_pairwise_correlation(
        returns[tickers].tail(config.correlation_long_lookback_days)
    )
    return {"short": short, "long": long, "shift": short - long}


def _average_pairwise_correlation(frame: pd.DataFrame) -> float:
    if frame.shape[1] < 2 or frame.shape[0] < 20:
        return 0.0
    corr = frame.corr().replace([np.inf, -np.inf], np.nan)
    if corr.empty:
        return 0.0
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    values = corr.where(mask).stack().dropna()
    return float(values.mean()) if not values.empty else 0.0


def _correlation_regime_frame(
    returns: pd.DataFrame,
    weights: pd.Series,
    snapshot: _MetricSnapshot,
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    if snapshot.average_correlation_short >= 0.65:
        regime = "high_correlation"
    elif snapshot.average_correlation_short >= 0.35:
        regime = "normal_correlation"
    else:
        regime = "diversified_correlation"
    if snapshot.correlation_shift >= config.correlation_shift_threshold:
        regime = "correlation_breakdown"
    return pd.DataFrame(
        [
            {
                "regime": regime,
                "average_correlation_short": snapshot.average_correlation_short,
                "average_correlation_long": snapshot.average_correlation_long,
                "correlation_shift": snapshot.correlation_shift,
                "short_lookback_days": config.correlation_short_lookback_days,
                "long_lookback_days": config.correlation_long_lookback_days,
                "active_positions": int((weights > 0.005).sum()),
                "interpretation": _correlation_interpretation(regime),
            }
        ]
    )


def _correlation_interpretation(regime: str) -> str:
    if regime == "correlation_breakdown":
        return (
            "Holdings are moving together more than usual; diversification may fail in a selloff."
        )
    if regime == "high_correlation":
        return "Holdings are highly correlated; size as a concentrated risk trade."
    if regime == "normal_correlation":
        return "Correlation is normal; diversification is present but not a left-tail hedge."
    return "Current holdings have relatively low pairwise correlation."


def _apply_constraints(
    returns: pd.DataFrame,
    target_weights: pd.Series,
    scenario_budget: _ScenarioBudget,
    config: PortfolioRiskConfig,
) -> tuple[pd.Series, tuple[str, ...]]:
    weights = target_weights.copy()
    applied: list[str] = []

    weights, capped = _cap_single_assets(weights, scenario_budget.max_single_asset_weight, config)
    if capped:
        applied.append("max_single_asset")

    weights, raised_defensive = _raise_defensive_weight(
        weights,
        scenario_budget.min_defensive_weight,
        config,
    )
    if raised_defensive:
        applied.append("scenario_min_defensive")

    for _ in range(4):
        snapshot = _metric_snapshot(returns, weights, scenario_budget, config)
        scalers: list[tuple[str, float]] = []
        if snapshot.equity_beta > scenario_budget.max_equity_beta > 0:
            scalers.append(("equity_beta", scenario_budget.max_equity_beta / snapshot.equity_beta))
        if snapshot.ai_beta > scenario_budget.max_ai_beta > 0:
            scalers.append(("ai_beta", scenario_budget.max_ai_beta / snapshot.ai_beta))
        if snapshot.expected_shortfall_95 > scenario_budget.max_expected_shortfall_95 > 0:
            scalers.append(
                (
                    "expected_shortfall",
                    scenario_budget.max_expected_shortfall_95 / snapshot.expected_shortfall_95,
                )
            )
        if snapshot.max_stress_loss > scenario_budget.max_stress_loss > 0:
            scalers.append(
                ("stress_loss", scenario_budget.max_stress_loss / snapshot.max_stress_loss)
            )
        if (
            snapshot.scenario_weighted_stress_loss
            > scenario_budget.max_scenario_weighted_stress_loss
            > 0
        ):
            scalers.append(
                (
                    "scenario_weighted_stress",
                    scenario_budget.max_scenario_weighted_stress_loss
                    / snapshot.scenario_weighted_stress_loss,
                )
            )
        if not scalers:
            break
        constraint, scaler = min(scalers, key=lambda item: item[1])
        scaler = float(np.clip(scaler, config.min_risk_asset_multiplier, 1.0))
        if scaler >= 0.999:
            break
        weights = _scale_risk_assets(weights, scaler, config)
        applied.append(constraint)

    weights, capped_after_scaling = _cap_single_assets(
        weights,
        scenario_budget.max_single_asset_weight,
        config,
    )
    if capped_after_scaling:
        applied.append("max_single_asset")
    weights = _normalize_weights(weights, config.defensive_ticker)
    return weights, tuple(dict.fromkeys(applied))


def _cap_single_assets(
    weights: pd.Series,
    max_weight: float,
    config: PortfolioRiskConfig,
) -> tuple[pd.Series, bool]:
    adjusted = weights.copy()
    if adjusted.empty:
        return adjusted, False
    capped = False
    freed_weight = 0.0
    for ticker, weight in adjusted.items():
        if ticker == config.defensive_ticker:
            continue
        if weight > max_weight:
            capped = True
            freed_weight += float(weight - max_weight)
            adjusted.loc[ticker] = max_weight
    if capped:
        adjusted.loc[config.defensive_ticker] = float(
            adjusted.get(config.defensive_ticker, 0.0)
        ) + (freed_weight)
    return _normalize_weights(adjusted, config.defensive_ticker), capped


def _raise_defensive_weight(
    weights: pd.Series,
    min_defensive_weight: float,
    config: PortfolioRiskConfig,
) -> tuple[pd.Series, bool]:
    adjusted = weights.copy()
    defensive_weight = float(adjusted.get(config.defensive_ticker, 0.0))
    if defensive_weight >= min_defensive_weight:
        return adjusted, False
    risk_tickers = [ticker for ticker in adjusted.index if ticker != config.defensive_ticker]
    risk_weight = float(adjusted.loc[risk_tickers].sum()) if risk_tickers else 0.0
    needed = min(min_defensive_weight - defensive_weight, risk_weight)
    if needed <= 0 or risk_weight <= 0:
        return adjusted, False
    scaler = max(0.0, 1.0 - needed / risk_weight)
    adjusted.loc[risk_tickers] = adjusted.loc[risk_tickers] * scaler
    adjusted.loc[config.defensive_ticker] = defensive_weight + needed
    return _normalize_weights(adjusted, config.defensive_ticker), True


def _scale_risk_assets(
    weights: pd.Series,
    scaler: float,
    config: PortfolioRiskConfig,
) -> pd.Series:
    adjusted = weights.copy()
    risk_tickers = [ticker for ticker in adjusted.index if ticker != config.defensive_ticker]
    if not risk_tickers:
        return adjusted
    previous_risk_weight = float(adjusted.loc[risk_tickers].sum())
    adjusted.loc[risk_tickers] = adjusted.loc[risk_tickers] * scaler
    new_risk_weight = float(adjusted.loc[risk_tickers].sum())
    adjusted.loc[config.defensive_ticker] = float(adjusted.get(config.defensive_ticker, 0.0)) + max(
        0.0,
        previous_risk_weight - new_risk_weight,
    )
    return _normalize_weights(adjusted, config.defensive_ticker)


def _scenario_budget_frame(scenario_budget: _ScenarioBudget) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "risk_off_probability": scenario_budget.risk_off_probability,
                "transition_probability": scenario_budget.transition_probability,
                "fragile_upside_probability": scenario_budget.fragile_upside_probability,
                "risk_on_probability": scenario_budget.risk_on_probability,
                "ai_unwind_probability": scenario_budget.ai_unwind_probability,
                "credit_stress_probability": scenario_budget.credit_stress_probability,
                "inflation_oil_probability": scenario_budget.inflation_oil_probability,
                "scenario_risk_multiplier": scenario_budget.scenario_risk_multiplier,
                "max_single_asset_weight": scenario_budget.max_single_asset_weight,
                "max_concentration_hhi": scenario_budget.max_concentration_hhi,
                "max_equity_beta": scenario_budget.max_equity_beta,
                "max_ai_beta": scenario_budget.max_ai_beta,
                "max_expected_shortfall_95": scenario_budget.max_expected_shortfall_95,
                "max_stress_loss": scenario_budget.max_stress_loss,
                "max_scenario_weighted_stress_loss": (
                    scenario_budget.max_scenario_weighted_stress_loss
                ),
                "min_defensive_weight": scenario_budget.min_defensive_weight,
            }
        ]
    )


def _summary_frame(
    pre_snapshot: _MetricSnapshot,
    post_snapshot: _MetricSnapshot,
    scenario_budget: _ScenarioBudget,
    applied_constraints: tuple[str, ...],
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    if (
        post_snapshot.max_stress_loss > scenario_budget.max_stress_loss
        or post_snapshot.expected_shortfall_95 > scenario_budget.max_expected_shortfall_95
        or post_snapshot.equity_beta > scenario_budget.max_equity_beta
        or post_snapshot.ai_beta > scenario_budget.max_ai_beta
    ):
        risk_level = "constraint_breach"
    elif applied_constraints:
        risk_level = "risk_reduced"
    elif post_snapshot.correlation_shift >= config.correlation_shift_threshold:
        risk_level = "watch_correlation_shift"
    else:
        risk_level = "within_limits"
    portfolio_risk_multiplier = (
        post_snapshot.risk_asset_weight / pre_snapshot.risk_asset_weight
        if pre_snapshot.risk_asset_weight > 0
        else 1.0
    )
    return pd.DataFrame(
        [
            {
                "portfolio_risk_level": risk_level,
                "scenario_risk_multiplier": scenario_budget.scenario_risk_multiplier,
                "portfolio_risk_multiplier": float(np.clip(portfolio_risk_multiplier, 0.0, 1.0)),
                "applied_constraints": ", ".join(applied_constraints) or "none",
                "pre_equity_beta": pre_snapshot.equity_beta,
                "post_equity_beta": post_snapshot.equity_beta,
                "max_equity_beta": scenario_budget.max_equity_beta,
                "pre_ai_beta": pre_snapshot.ai_beta,
                "post_ai_beta": post_snapshot.ai_beta,
                "max_ai_beta": scenario_budget.max_ai_beta,
                "pre_expected_shortfall_95": pre_snapshot.expected_shortfall_95,
                "post_expected_shortfall_95": post_snapshot.expected_shortfall_95,
                "max_expected_shortfall_95": scenario_budget.max_expected_shortfall_95,
                "pre_max_stress_loss": pre_snapshot.max_stress_loss,
                "post_max_stress_loss": post_snapshot.max_stress_loss,
                "max_stress_loss": scenario_budget.max_stress_loss,
                "pre_scenario_weighted_stress_loss": pre_snapshot.scenario_weighted_stress_loss,
                "post_scenario_weighted_stress_loss": (post_snapshot.scenario_weighted_stress_loss),
                "max_scenario_weighted_stress_loss": (
                    scenario_budget.max_scenario_weighted_stress_loss
                ),
                "post_defensive_weight": post_snapshot.defensive_weight,
                "min_defensive_weight": scenario_budget.min_defensive_weight,
                "correlation_regime_shift": post_snapshot.correlation_shift,
            }
        ]
    )


def _beta_decomposition_frame(
    pre_factor_exposures: pd.DataFrame,
    post_factor_exposures: pd.DataFrame,
) -> pd.DataFrame:
    if pre_factor_exposures.empty and post_factor_exposures.empty:
        return pd.DataFrame()
    pre = pre_factor_exposures[["factor", "proxy_ticker", "beta"]].rename(
        columns={"beta": "pre_beta"}
    )
    post = post_factor_exposures[["factor", "proxy_ticker", "beta"]].rename(
        columns={"beta": "post_beta"}
    )
    merged = pd.merge(pre, post, on=["factor", "proxy_ticker"], how="outer")
    merged["beta_change"] = merged["post_beta"] - merged["pre_beta"]
    absolute_total = float(merged["post_beta"].abs().sum())
    merged["post_absolute_beta_share"] = (
        merged["post_beta"].abs() / absolute_total if absolute_total > 0 else 0.0
    )
    return merged.sort_values("post_absolute_beta_share", ascending=False)


def _marginal_risk_contribution(
    returns: pd.DataFrame,
    weights: pd.Series,
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    positive = weights[weights > 0.005]
    tickers = [ticker for ticker in positive.index if ticker in returns]
    if not tickers:
        return pd.DataFrame()
    if len(tickers) == 1:
        ticker = tickers[0]
        return pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "weight": float(positive.loc[ticker]),
                    "risk_contribution_pct": 1.0,
                    "annualized_vol_contribution": float(
                        returns[ticker].tail(config.covariance_lookback_days).std()
                        * np.sqrt(TRADING_DAYS_PER_YEAR)
                        * positive.loc[ticker]
                    ),
                }
            ]
        )
    covariance_returns = returns[tickers].tail(config.covariance_lookback_days).dropna(how="all")
    if covariance_returns.shape[0] < 20:
        return pd.DataFrame()
    covariance = covariance_returns.cov()
    weight_vector = positive.reindex(tickers).astype(float)
    covariance_matrix = covariance.reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy()
    weights_array = weight_vector.to_numpy()
    portfolio_variance = float(weights_array.T @ covariance_matrix @ weights_array)
    if portfolio_variance <= 0:
        return pd.DataFrame()
    marginal = covariance_matrix @ weights_array
    component = weights_array * marginal / portfolio_variance
    portfolio_vol = float(np.sqrt(portfolio_variance) * np.sqrt(TRADING_DAYS_PER_YEAR))
    frame = pd.DataFrame(
        {
            "ticker": tickers,
            "weight": weight_vector.values,
            "risk_contribution_pct": component,
            "annualized_vol_contribution": component * portfolio_vol,
        }
    )
    return frame.sort_values("risk_contribution_pct", ascending=False)


def _constraint_report_frame(
    pre_snapshot: _MetricSnapshot,
    post_snapshot: _MetricSnapshot,
    scenario_budget: _ScenarioBudget,
    current_weights: pd.Series,
    risk_adjusted_weights: pd.Series,
    applied_constraints: tuple[str, ...],
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    turnover = _turnover(current_weights, risk_adjusted_weights)
    rows = [
        _max_constraint_row(
            "max_single_asset",
            pre_snapshot.max_single_asset_weight,
            post_snapshot.max_single_asset_weight,
            scenario_budget.max_single_asset_weight,
            applied_constraints,
        ),
        _max_constraint_row(
            "concentration_hhi",
            pre_snapshot.concentration_hhi,
            post_snapshot.concentration_hhi,
            scenario_budget.max_concentration_hhi,
            applied_constraints,
            hard=False,
        ),
        _min_constraint_row(
            "min_defensive_weight",
            pre_snapshot.defensive_weight,
            post_snapshot.defensive_weight,
            scenario_budget.min_defensive_weight,
            applied_constraints,
        ),
        _max_constraint_row(
            "equity_beta",
            pre_snapshot.equity_beta,
            post_snapshot.equity_beta,
            scenario_budget.max_equity_beta,
            applied_constraints,
        ),
        _max_constraint_row(
            "ai_beta",
            pre_snapshot.ai_beta,
            post_snapshot.ai_beta,
            scenario_budget.max_ai_beta,
            applied_constraints,
        ),
        _max_constraint_row(
            "expected_shortfall_95",
            pre_snapshot.expected_shortfall_95,
            post_snapshot.expected_shortfall_95,
            scenario_budget.max_expected_shortfall_95,
            applied_constraints,
        ),
        _max_constraint_row(
            "max_stress_loss",
            pre_snapshot.max_stress_loss,
            post_snapshot.max_stress_loss,
            scenario_budget.max_stress_loss,
            applied_constraints,
        ),
        _max_constraint_row(
            "scenario_weighted_stress_loss",
            pre_snapshot.scenario_weighted_stress_loss,
            post_snapshot.scenario_weighted_stress_loss,
            scenario_budget.max_scenario_weighted_stress_loss,
            applied_constraints,
        ),
        {
            "constraint": "max_turnover_soft_guardrail",
            "pre_value": turnover,
            "post_value": turnover,
            "limit": config.max_turnover,
            "status": "watch" if turnover > config.max_turnover else "ok",
            "hard_constraint": False,
            "action": (
                "Review trade staging; this is a human-executed turnover warning."
                if turnover > config.max_turnover
                else "No action."
            ),
        },
        {
            "constraint": "correlation_regime_shift_watch",
            "pre_value": pre_snapshot.correlation_shift,
            "post_value": post_snapshot.correlation_shift,
            "limit": config.correlation_shift_threshold,
            "status": (
                "watch"
                if post_snapshot.correlation_shift >= config.correlation_shift_threshold
                else "ok"
            ),
            "hard_constraint": False,
            "action": (
                "Treat diversification as less reliable until pairwise correlation normalizes."
                if post_snapshot.correlation_shift >= config.correlation_shift_threshold
                else "No action."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _max_constraint_row(
    constraint: str,
    pre_value: float,
    post_value: float,
    limit: float,
    applied_constraints: tuple[str, ...],
    *,
    hard: bool = True,
) -> dict[str, object]:
    if post_value > limit:
        status = "breach" if hard else "watch"
    elif pre_value > limit or constraint in applied_constraints:
        status = "adjusted"
    else:
        status = "ok"
    return {
        "constraint": constraint,
        "pre_value": pre_value,
        "post_value": post_value,
        "limit": limit,
        "status": status,
        "hard_constraint": hard,
        "action": _constraint_action(status, constraint),
    }


def _min_constraint_row(
    constraint: str,
    pre_value: float,
    post_value: float,
    limit: float,
    applied_constraints: tuple[str, ...],
) -> dict[str, object]:
    if post_value < limit:
        status = "breach"
    elif (
        pre_value < limit
        or constraint in applied_constraints
        or "scenario_min_defensive" in applied_constraints
    ):
        status = "adjusted"
    else:
        status = "ok"
    return {
        "constraint": constraint,
        "pre_value": pre_value,
        "post_value": post_value,
        "limit": limit,
        "status": status,
        "hard_constraint": True,
        "action": _constraint_action(status, constraint),
    }


def _constraint_action(status: str, constraint: str) -> str:
    if status == "breach":
        return f"Still above the {constraint} limit after risk scaling; manual review required."
    if status == "adjusted":
        return f"Risk engine adjusted sizing for {constraint}."
    if status == "watch":
        return f"Watch {constraint}; not a hard sizing block."
    return "No action."


def _sizing_adjustments_frame(
    current_weights: pd.Series,
    pre_weights: pd.Series,
    post_weights: pd.Series,
    applied_constraints: tuple[str, ...],
    config: PortfolioRiskConfig,
) -> pd.DataFrame:
    tickers = sorted(set(current_weights.index) | set(pre_weights.index) | set(post_weights.index))
    rows: list[dict[str, object]] = []
    reason = ", ".join(applied_constraints) or "no risk-engine adjustment"
    for ticker in tickers:
        current = float(current_weights.get(ticker, 0.0))
        pre = float(pre_weights.get(ticker, 0.0))
        post = float(post_weights.get(ticker, 0.0))
        delta_from_current = post - current
        risk_engine_delta = post - pre
        if abs(delta_from_current) < 0.02:
            action = "HOLD"
        elif delta_from_current > 0:
            action = "ADD"
        else:
            action = "REDUCE"
        rows.append(
            {
                "ticker": ticker,
                "group": _ticker_group(ticker),
                "current_weight": current,
                "pre_risk_target_weight": pre,
                "risk_adjusted_weight": post,
                "delta_weight": delta_from_current,
                "risk_engine_delta": risk_engine_delta,
                "action": action,
                "risk_adjustment_reason": (
                    "absorbs reduced risk weight"
                    if ticker == config.defensive_ticker and risk_engine_delta > 0
                    else reason
                ),
            }
        )
    frame = pd.DataFrame(rows)
    material = (
        frame[
            [
                "current_weight",
                "pre_risk_target_weight",
                "risk_adjusted_weight",
                "delta_weight",
                "risk_engine_delta",
            ]
        ]
        .abs()
        .max(axis=1)
        >= 0.005
    )
    return frame[material].sort_values("delta_weight")


def _turnover(current_weights: pd.Series, target_weights: pd.Series) -> float:
    tickers = sorted(set(current_weights.index) | set(target_weights.index))
    current = current_weights.reindex(tickers).fillna(0.0)
    target = target_weights.reindex(tickers).fillna(0.0)
    return float((target - current).abs().sum())


def _group_weight(weights: pd.Series, group: str) -> float:
    return float(
        sum(weight for ticker, weight in weights.items() if _ticker_group(str(ticker)) == group)
    )
