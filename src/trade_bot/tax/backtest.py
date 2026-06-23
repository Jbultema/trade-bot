from __future__ import annotations

import uuid
from dataclasses import dataclass

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.tax.account import TaxAccountProfile
from trade_bot.tax.lots import EPSILON, TaxLotLedger


@dataclass(frozen=True)
class TaxBacktestResult:
    name: str
    account_profile: TaxAccountProfile
    pre_tax_equity: pd.Series
    after_tax_equity: pd.Series
    after_tax_returns: pd.Series
    annual_tax_summary: pd.DataFrame
    realized_lots: pd.DataFrame
    open_lots: pd.DataFrame
    summary: pd.Series


def simulate_taxable_backtest(
    result: BacktestResult,
    prices: pd.DataFrame,
    profile: TaxAccountProfile | None = None,
    *,
    substitute_map: dict[str, list[str]] | None = None,
) -> TaxBacktestResult:
    profile = profile or TaxAccountProfile()
    aligned_prices = prices.reindex(
        index=result.weights.index, columns=result.weights.columns
    ).ffill()
    ledger = TaxLotLedger(profile)
    if profile.is_taxable:
        _reconstruct_executions(result, aligned_prices, ledger)
        ledger.apply_wash_sale_rules(substitute_map=substitute_map)
    realized = ledger.realized_lots_frame()
    open_lots = ledger.open_lots_frame()
    annual = _annual_tax_summary(realized, result.equity.index, profile)
    after_tax_equity = _apply_tax_cash_flows(result.equity, annual)
    after_tax_returns = after_tax_equity.pct_change().fillna(
        result.returns.reindex(after_tax_equity.index).fillna(0.0)
    )
    pre_tax_metrics = calculate_metrics(
        result.name,
        result.returns,
        result.equity,
        result.turnover,
        result.transaction_costs,
    )
    after_tax_metrics = calculate_metrics(
        result.name,
        after_tax_returns,
        after_tax_equity,
        result.turnover.reindex(after_tax_returns.index).fillna(0.0),
        result.transaction_costs.reindex(after_tax_returns.index).fillna(0.0),
    )
    summary = _summary_series(
        result.name,
        profile,
        pre_tax_metrics,
        after_tax_metrics,
        annual,
        realized,
    )
    return TaxBacktestResult(
        name=result.name,
        account_profile=profile,
        pre_tax_equity=result.equity,
        after_tax_equity=after_tax_equity,
        after_tax_returns=after_tax_returns,
        annual_tax_summary=annual,
        realized_lots=realized,
        open_lots=open_lots,
        summary=summary,
    )


def tax_metrics_frame(results: list[TaxBacktestResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([result.summary.to_dict() for result in results]).set_index("strategy")


def _reconstruct_executions(
    result: BacktestResult,
    prices: pd.DataFrame,
    ledger: TaxLotLedger,
) -> None:
    tickers = [ticker for ticker in result.weights.columns if ticker in prices.columns]
    if not tickers:
        return
    positions = pd.Series(0.0, index=tickers, dtype=float)
    weights = result.weights.reindex(columns=tickers).fillna(0.0)
    for date, weight_row in weights.iterrows():
        if date not in prices.index or date not in result.equity.index:
            continue
        price_row = prices.loc[date, tickers]
        equity = float(result.equity.loc[date])
        if equity <= 0:
            continue
        current_values = positions * price_row.astype(float)
        target_values = weight_row.astype(float).clip(lower=0.0) * equity
        deltas = (target_values - current_values).dropna()
        for ticker, delta in deltas[deltas < -EPSILON].sort_values().items():
            price = float(price_row[ticker])
            if price <= 0 or pd.isna(price):
                continue
            sell_quantity = min(float(positions[ticker]), abs(float(delta)) / price)
            if sell_quantity <= EPSILON:
                continue
            ledger.process_execution(
                execution_id=_execution_id(result.name, date, ticker, "SELL"),
                mode="backtest",
                account=result.name,
                ticker=ticker,
                side="SELL",
                quantity=sell_quantity,
                price=price,
                executed_at=date,
            )
            positions[ticker] = max(0.0, positions[ticker] - sell_quantity)
        for ticker, delta in deltas[deltas > EPSILON].sort_values(ascending=False).items():
            price = float(price_row[ticker])
            if price <= 0 or pd.isna(price):
                continue
            buy_quantity = float(delta) / price
            if buy_quantity <= EPSILON:
                continue
            ledger.process_execution(
                execution_id=_execution_id(result.name, date, ticker, "BUY"),
                mode="backtest",
                account=result.name,
                ticker=ticker,
                side="BUY",
                quantity=buy_quantity,
                price=price,
                executed_at=date,
            )
            positions[ticker] += buy_quantity


def _annual_tax_summary(
    realized: pd.DataFrame,
    equity_index: pd.Index,
    profile: TaxAccountProfile,
) -> pd.DataFrame:
    if len(equity_index) == 0:
        return pd.DataFrame()
    years = range(
        int(pd.Timestamp(equity_index.min()).year), int(pd.Timestamp(equity_index.max()).year) + 1
    )
    if realized.empty or not profile.is_taxable:
        return pd.DataFrame(
            [
                _annual_row(
                    year,
                    short_term_gain=0.0,
                    long_term_gain=0.0,
                    realized_loss_harvested=0.0,
                    wash_sale_disallowed_loss=0.0,
                    tax_liability=0.0,
                    tax_benefit=0.0,
                    loss_carryforward_end=0.0,
                )
                for year in years
            ]
        )
    frame = realized.copy()
    frame["sold_at"] = pd.to_datetime(frame["sold_at"])
    frame["year"] = frame["sold_at"].dt.year
    carryforward = profile.starting_loss_carryforward
    rows = []
    for year in years:
        year_frame = frame[frame["year"] == year]
        taxable = year_frame.get("taxable_gain_loss", pd.Series(dtype=float)).astype(float)
        realized_gl = year_frame.get("realized_gain_loss", pd.Series(dtype=float)).astype(float)
        short_term_gain = (
            float(taxable[year_frame.get("term") == "short"].sum()) if not year_frame.empty else 0.0
        )
        long_term_gain = (
            float(taxable[year_frame.get("term") == "long"].sum()) if not year_frame.empty else 0.0
        )
        realized_loss_harvested = (
            float((-realized_gl.clip(upper=0.0)).sum()) if not year_frame.empty else 0.0
        )
        wash_sale_disallowed_loss = (
            float(
                year_frame.get("wash_sale_disallowed_loss", pd.Series(dtype=float))
                .astype(float)
                .sum()
            )
            if not year_frame.empty
            else 0.0
        )
        positive_short = max(0.0, short_term_gain)
        positive_long = max(0.0, long_term_gain)
        available_losses = carryforward + max(0.0, -short_term_gain) + max(0.0, -long_term_gain)
        short_offset = min(positive_short, available_losses)
        positive_short -= short_offset
        available_losses -= short_offset
        long_offset = min(positive_long, available_losses)
        positive_long -= long_offset
        available_losses -= long_offset
        tax_liability = (
            positive_short * profile.short_term_tax_rate
            + positive_long * profile.long_term_tax_rate
        )
        ordinary_loss = min(available_losses, profile.annual_loss_deduction_limit)
        tax_benefit = ordinary_loss * profile.short_term_tax_rate
        carryforward = max(0.0, available_losses - ordinary_loss)
        rows.append(
            _annual_row(
                year,
                short_term_gain=short_term_gain,
                long_term_gain=long_term_gain,
                realized_loss_harvested=realized_loss_harvested,
                wash_sale_disallowed_loss=wash_sale_disallowed_loss,
                tax_liability=tax_liability,
                tax_benefit=tax_benefit,
                loss_carryforward_end=carryforward,
            )
        )
    return pd.DataFrame(rows)


def _annual_row(
    year: int,
    *,
    short_term_gain: float,
    long_term_gain: float,
    realized_loss_harvested: float,
    wash_sale_disallowed_loss: float,
    tax_liability: float,
    tax_benefit: float,
    loss_carryforward_end: float,
) -> dict[str, float | int]:
    return {
        "year": year,
        "realized_short_term_gain": short_term_gain,
        "realized_long_term_gain": long_term_gain,
        "realized_loss_harvested": realized_loss_harvested,
        "wash_sale_disallowed_loss": wash_sale_disallowed_loss,
        "tax_liability": tax_liability,
        "tax_benefit": tax_benefit,
        "tax_cash_flow": tax_benefit - tax_liability,
        "loss_carryforward_end": loss_carryforward_end,
    }


def _apply_tax_cash_flows(equity: pd.Series, annual: pd.DataFrame) -> pd.Series:
    if annual.empty:
        return equity.copy()
    cash_flows = pd.Series(0.0, index=equity.index, dtype=float)
    for _, row in annual.iterrows():
        year_dates = equity.index[pd.DatetimeIndex(equity.index).year == int(row["year"])]
        if len(year_dates) == 0:
            continue
        cash_flows.loc[year_dates[-1]] += float(row["tax_cash_flow"])
    after_tax = equity + cash_flows.cumsum()
    return after_tax.clip(lower=max(1.0, float(equity.iloc[0]) * 0.01)).rename(equity.name)


def _summary_series(
    name: str,
    profile: TaxAccountProfile,
    pre_tax_metrics: object,
    after_tax_metrics: object,
    annual: pd.DataFrame,
    realized: pd.DataFrame,
) -> pd.Series:
    total_tax_liability = float(annual.get("tax_liability", pd.Series(dtype=float)).sum())
    total_tax_benefit = float(annual.get("tax_benefit", pd.Series(dtype=float)).sum())
    total_realized = (
        float(realized.get("realized_gain_loss", pd.Series(dtype=float)).abs().sum())
        if not realized.empty
        else 0.0
    )
    short_realized = (
        float(realized.loc[realized["term"] == "short", "realized_gain_loss"].abs().sum())
        if not realized.empty and "term" in realized
        else 0.0
    )
    short_term_gain_share = short_realized / total_realized if total_realized > EPSILON else 0.0
    pre_cagr = float(pre_tax_metrics.cagr)
    after_cagr = float(after_tax_metrics.cagr)
    return pd.Series(
        {
            "strategy": name,
            "tax_model_status": (
                "taxable_estimated" if profile.is_taxable else "pre_tax_or_tax_deferred"
            ),
            "tax_account_type": profile.account_type,
            "after_tax_final_equity": float(after_tax_metrics.final_equity),
            "after_tax_cagr": after_cagr,
            "after_tax_max_drawdown": float(after_tax_metrics.max_drawdown),
            "after_tax_calmar": float(after_tax_metrics.calmar),
            "tax_drag_bps_per_year": (pre_cagr - after_cagr) * 10000.0,
            "total_tax_liability": total_tax_liability,
            "total_tax_benefit": total_tax_benefit,
            "net_estimated_tax_paid": total_tax_liability - total_tax_benefit,
            "realized_short_term_gain": float(
                annual.get("realized_short_term_gain", pd.Series(dtype=float)).sum()
            ),
            "realized_long_term_gain": float(
                annual.get("realized_long_term_gain", pd.Series(dtype=float)).sum()
            ),
            "realized_loss_harvested": float(
                annual.get("realized_loss_harvested", pd.Series(dtype=float)).sum()
            ),
            "wash_sale_disallowed_loss": float(
                annual.get("wash_sale_disallowed_loss", pd.Series(dtype=float)).sum()
            ),
            "loss_carryforward_end": (
                float(annual["loss_carryforward_end"].iloc[-1]) if not annual.empty else 0.0
            ),
            "short_term_gain_share": short_term_gain_share,
        }
    )


def _execution_id(name: str, date: pd.Timestamp, ticker: str, side: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}:{date.isoformat()}:{ticker}:{side}"))
