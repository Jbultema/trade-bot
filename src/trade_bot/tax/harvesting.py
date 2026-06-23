from __future__ import annotations

import pandas as pd

from trade_bot.tax.account import TaxLossHarvestConfig


def find_tax_loss_harvest_candidates(
    open_lots: pd.DataFrame,
    current_prices: dict[str, float] | pd.Series,
    config: TaxLossHarvestConfig | None = None,
    *,
    substitute_map: dict[str, list[str]] | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    cfg = config or TaxLossHarvestConfig()
    if open_lots.empty:
        return _empty_frame()
    prices = pd.Series(current_prices, dtype=float)
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.utcnow().tz_localize(None)
    rows = []
    for _, row in open_lots.iterrows():
        ticker = str(row["ticker"]).upper()
        if ticker not in prices or pd.isna(prices[ticker]) or float(prices[ticker]) <= 0:
            continue
        remaining_quantity = float(row["remaining_quantity"])
        cost_basis_per_share = float(row["cost_basis_per_share"])
        current_price = float(prices[ticker])
        unrealized_gain_loss = (current_price - cost_basis_per_share) * remaining_quantity
        cost_basis = cost_basis_per_share * remaining_quantity
        loss_pct = unrealized_gain_loss / cost_basis if cost_basis > 0 else 0.0
        if unrealized_gain_loss > -cfg.min_loss_amount:
            continue
        if loss_pct > -cfg.min_loss_pct:
            continue
        acquired_at = pd.Timestamp(row["acquired_at"])
        days_held = max(0, (as_of_ts - acquired_at).days)
        rows.append(
            {
                "lot_id": row["lot_id"],
                "mode": row.get("mode", ""),
                "account": row.get("account", ""),
                "ticker": ticker,
                "acquired_at": acquired_at.isoformat(),
                "days_held": days_held,
                "quantity": remaining_quantity,
                "cost_basis": cost_basis,
                "current_value": current_price * remaining_quantity,
                "current_price": current_price,
                "unrealized_gain_loss": unrealized_gain_loss,
                "unrealized_loss_pct": loss_pct,
                "substitute_candidates": (
                    ", ".join(substitute_map.get(ticker, [])) if substitute_map else ""
                ),
                "cooldown_days": cfg.cooldown_days,
                "wash_sale_window_days": cfg.wash_sale_window_days,
                "wash_sale_enforcement": cfg.wash_sale_enforcement,
                "status": "human_review_required",
            }
        )
    if not rows:
        return _empty_frame()
    return pd.DataFrame(rows).sort_values("unrealized_gain_loss")


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "lot_id",
            "mode",
            "account",
            "ticker",
            "acquired_at",
            "days_held",
            "quantity",
            "cost_basis",
            "current_value",
            "current_price",
            "unrealized_gain_loss",
            "unrealized_loss_pct",
            "substitute_candidates",
            "cooldown_days",
            "wash_sale_window_days",
            "wash_sale_enforcement",
            "status",
        ]
    )
