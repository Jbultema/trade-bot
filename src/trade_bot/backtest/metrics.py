from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.features.indicators import TRADING_DAYS_PER_YEAR, drawdown


@dataclass(frozen=True)
class PerformanceMetrics:
    name: str
    start: str
    end: str
    years: float
    final_equity: float
    cagr: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    best_day: float
    worst_day: float
    average_turnover: float
    total_transaction_cost: float
    historical_evidence_basis: str


def calculate_metrics(
    name: str,
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    transaction_costs: pd.Series,
    *,
    historical_evidence_basis: str = "modern_universe_replay",
) -> PerformanceMetrics:
    clean_returns = returns.dropna()
    clean_equity = equity.loc[clean_returns.index]
    if clean_returns.empty:
        raise ValueError("Cannot calculate metrics for an empty return series.")

    years = max((clean_returns.index[-1] - clean_returns.index[0]).days / 365.25, 1 / 365.25)
    final_equity = float(clean_equity.iloc[-1])
    first_growth = 1.0 + float(clean_returns.iloc[0])
    initial_equity = (
        float(clean_equity.iloc[0] / first_growth)
        if abs(first_growth) > 1e-12
        else float(clean_equity.iloc[0])
    )
    cagr = (final_equity / initial_equity) ** (1.0 / years) - 1.0
    annualized_vol = float(clean_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = _safe_ratio(clean_returns.mean() * TRADING_DAYS_PER_YEAR, annualized_vol)

    downside = clean_returns.clip(upper=0.0)
    downside_vol = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino = _safe_ratio(clean_returns.mean() * TRADING_DAYS_PER_YEAR, downside_vol)

    max_dd = float(drawdown(clean_equity).min())
    calmar = _safe_ratio(cagr, abs(max_dd))

    return PerformanceMetrics(
        name=name,
        start=str(clean_returns.index[0].date()),
        end=str(clean_returns.index[-1].date()),
        years=years,
        final_equity=final_equity,
        cagr=float(cagr),
        annualized_volatility=annualized_vol,
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=max_dd,
        calmar=float(calmar),
        best_day=float(clean_returns.max()),
        worst_day=float(clean_returns.min()),
        average_turnover=float(turnover.reindex(clean_returns.index).mean()),
        total_transaction_cost=float(transaction_costs.reindex(clean_returns.index).sum()),
        historical_evidence_basis=historical_evidence_basis,
    )


def metrics_frame(metrics: list[PerformanceMetrics]) -> pd.DataFrame:
    return pd.DataFrame([metric.__dict__ for metric in metrics]).set_index("name")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator
