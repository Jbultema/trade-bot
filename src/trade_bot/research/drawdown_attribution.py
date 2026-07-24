from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.features.indicators import bounded_forward_fill, daily_returns, drawdown


@dataclass(frozen=True)
class DrawdownAttribution:
    summary: pd.DataFrame
    contributors: pd.DataFrame
    exposure_path: pd.DataFrame


def build_drawdown_attribution(
    result: BacktestResult,
    prices: pd.DataFrame,
    *,
    defensive_ticker: str | None = "BIL",
    benchmarks: tuple[str, ...] = ("SPY", "QQQ"),
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    recovery_horizon_sessions: int = 63,
) -> DrawdownAttribution:
    equity = result.equity.sort_index().dropna().astype(float)
    if start is not None:
        equity = equity.loc[pd.Timestamp(start) :]
    if end is not None:
        equity = equity.loc[: pd.Timestamp(end)]
    if equity.empty:
        return DrawdownAttribution(
            summary=pd.DataFrame(),
            contributors=pd.DataFrame(),
            exposure_path=pd.DataFrame(),
        )

    drawdowns = drawdown(equity)
    trough_date = pd.Timestamp(drawdowns.idxmin())
    peak_date = pd.Timestamp(equity.loc[:trough_date].idxmax())
    peak_value = float(equity.loc[peak_date])
    recovery_candidates = equity.loc[trough_date:]
    recovery_candidates = recovery_candidates[recovery_candidates.ge(peak_value)]
    recovery_date = (
        pd.Timestamp(recovery_candidates.index[0]) if not recovery_candidates.empty else pd.NaT
    )
    trough_position = int(equity.index.get_loc(trough_date))
    fallback_position = min(
        len(equity.index) - 1,
        trough_position + recovery_horizon_sessions,
    )
    recovery_end = (
        recovery_date if pd.notna(recovery_date) else pd.Timestamp(equity.index[fallback_position])
    )

    weights = result.weights.reindex(equity.index).fillna(0.0).astype(float).clip(lower=0.0)
    interval_index = equity.loc[peak_date:trough_date].index
    return_interval_index = interval_index[1:]
    asset_returns = (
        daily_returns(prices)
        .reindex(
            index=return_interval_index,
            columns=weights.columns,
        )
        .fillna(0.0)
    )
    interval_weights = weights.reindex(return_interval_index).fillna(0.0)
    contribution = (interval_weights * asset_returns).sum(axis=0)
    contributors = pd.DataFrame(
        {
            "asset": contribution.index.astype(str),
            "gross_return_contribution": contribution.to_numpy(dtype=float),
            "average_weight": interval_weights.mean(axis=0).to_numpy(dtype=float),
            "maximum_weight": interval_weights.max(axis=0).to_numpy(dtype=float),
        }
    ).sort_values("gross_return_contribution")
    contributors["loss_rank"] = np.arange(1, len(contributors) + 1)

    risk_columns = [column for column in weights.columns if column != defensive_ticker]
    exposure = pd.DataFrame(index=equity.loc[peak_date:recovery_end].index)
    exposure["risk_assets"] = weights[risk_columns].sum(axis=1).reindex(exposure.index)
    exposure["defensive"] = (
        weights[defensive_ticker].reindex(exposure.index)
        if defensive_ticker and defensive_ticker in weights
        else 0.0
    )
    exposure["cash_or_unallocated"] = (1.0 - exposure["risk_assets"] - exposure["defensive"]).clip(
        lower=0.0
    )
    exposure.index.name = "market_date"
    exposure = exposure.reset_index()

    summary_row: dict[str, object] = {
        "strategy": result.name,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "recovery_measure_end": recovery_end,
        "max_drawdown": float(drawdowns.loc[trough_date]),
        "peak_to_trough_sessions": max(0, len(interval_index) - 1),
        "recovery_sessions": (
            max(0, len(equity.loc[trough_date:recovery_date]) - 1)
            if pd.notna(recovery_date)
            else np.nan
        ),
        "strategy_peak_to_trough_return": float(
            equity.loc[trough_date] / equity.loc[peak_date] - 1.0
        ),
        "strategy_recovery_period_return": float(
            equity.loc[recovery_end] / equity.loc[trough_date] - 1.0
        ),
        "turnover_peak_to_trough": float(result.turnover.reindex(interval_index).fillna(0.0).sum()),
        "transaction_cost_peak_to_trough": float(
            result.transaction_costs.reindex(interval_index).fillna(0.0).sum()
        ),
        "average_risk_exposure": float(exposure["risk_assets"].mean()),
        "minimum_risk_exposure": float(exposure["risk_assets"].min()),
        "average_defensive_exposure": float(exposure["defensive"].mean()),
    }
    filled_prices = bounded_forward_fill(prices.sort_index())
    for benchmark in benchmarks:
        if benchmark not in filled_prices:
            continue
        series = filled_prices[benchmark].dropna()
        if not {peak_date, trough_date, recovery_end}.issubset(series.index):
            continue
        peak_to_trough = float(series.loc[trough_date] / series.loc[peak_date] - 1.0)
        recovery_return = float(series.loc[recovery_end] / series.loc[trough_date] - 1.0)
        summary_row[f"{benchmark.lower()}_peak_to_trough_return"] = peak_to_trough
        summary_row[f"{benchmark.lower()}_recovery_period_return"] = recovery_return
        summary_row[f"missed_{benchmark.lower()}_recovery"] = recovery_return - float(
            summary_row["strategy_recovery_period_return"]
        )
    return DrawdownAttribution(
        summary=pd.DataFrame([summary_row]),
        contributors=contributors.reset_index(drop=True),
        exposure_path=exposure,
    )
