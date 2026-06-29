from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd

from trade_bot.DEFAULTS import DEFAULT_TAX_LOT_QUANTITY_EPSILON
from trade_bot.tax.account import TaxAccountProfile

EPSILON = DEFAULT_TAX_LOT_QUANTITY_EPSILON


@dataclass
class TaxLot:
    lot_id: str
    mode: str
    account: str
    ticker: str
    acquired_at: pd.Timestamp
    quantity: float
    remaining_quantity: float
    price: float
    cost_basis_per_share: float
    source_execution_id: str
    fees: float = 0.0
    wash_sale_adjustment: float = 0.0

    @property
    def total_cost_basis(self) -> float:
        return self.remaining_quantity * self.cost_basis_per_share


@dataclass
class RealizedTaxLot:
    realized_id: str
    mode: str
    account: str
    ticker: str
    acquired_at: pd.Timestamp
    sold_at: pd.Timestamp
    quantity: float
    proceeds: float
    cost_basis: float
    realized_gain_loss: float
    term: str
    source_lot_id: str
    source_execution_id: str
    sell_execution_id: str
    wash_sale_disallowed_loss: float = 0.0
    taxable_gain_loss: float | None = None
    wash_sale_status: str = "clear"

    def __post_init__(self) -> None:
        if self.taxable_gain_loss is None:
            self.taxable_gain_loss = self.realized_gain_loss + self.wash_sale_disallowed_loss


class TaxLotLedger:
    """Derived lot ledger for estimated backtests and journal executions."""

    def __init__(self, profile: TaxAccountProfile | None = None) -> None:
        self.profile = profile or TaxAccountProfile()
        self._lots: list[TaxLot] = []
        self._realized: list[RealizedTaxLot] = []

    @property
    def lots(self) -> tuple[TaxLot, ...]:
        return tuple(self._lots)

    @property
    def realized_lots(self) -> tuple[RealizedTaxLot, ...]:
        return tuple(self._realized)

    def process_execution(
        self,
        *,
        execution_id: str,
        mode: str,
        account: str,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        executed_at: str | pd.Timestamp,
        fees: float = 0.0,
    ) -> list[RealizedTaxLot]:
        if quantity <= 0:
            raise ValueError("Execution quantity must be positive.")
        if price <= 0:
            raise ValueError("Execution price must be positive.")
        timestamp = _timestamp(executed_at)
        side = side.upper()
        ticker = ticker.upper()
        if side == "BUY":
            self._add_lot(
                execution_id=execution_id,
                mode=mode,
                account=account,
                ticker=ticker,
                quantity=quantity,
                price=price,
                acquired_at=timestamp,
                fees=fees,
            )
            return []
        if side != "SELL":
            raise ValueError(f"Unsupported execution side: {side}")
        return self._sell_lots(
            execution_id=execution_id,
            mode=mode,
            account=account,
            ticker=ticker,
            quantity=quantity,
            price=price,
            sold_at=timestamp,
            fees=fees,
        )

    def process_frame(self, executions: pd.DataFrame) -> None:
        if executions.empty:
            return
        frame = executions.sort_values(["executed_at_utc", "created_at_utc"], na_position="last")
        for _, row in frame.iterrows():
            self.process_execution(
                execution_id=str(row.get("execution_id", "")),
                mode=str(row.get("mode", "")),
                account=str(row.get("account", "")),
                ticker=str(row["ticker"]),
                side=str(row["side"]),
                quantity=float(row["quantity"]),
                price=float(row["price"]),
                executed_at=row.get("executed_at_utc", row.get("executed_at")),
                fees=float(row.get("fees", 0.0) or 0.0),
            )

    def apply_wash_sale_rules(
        self,
        *,
        substitute_map: dict[str, list[str]] | None = None,
    ) -> None:
        if not self.profile.is_taxable or self.profile.wash_sale_enforcement == "off":
            return
        window = self.profile.wash_sale_window_days
        for realized in self._realized:
            if realized.realized_gain_loss >= -EPSILON:
                continue
            similar = _similar_tickers(realized.ticker, substitute_map)
            replacement_quantity = 0.0
            for lot in self._lots:
                if lot.lot_id == realized.source_lot_id:
                    continue
                if lot.remaining_quantity <= EPSILON:
                    continue
                if lot.ticker not in similar:
                    continue
                days = abs((lot.acquired_at - realized.sold_at).days)
                if days <= window:
                    replacement_quantity += lot.remaining_quantity
            if replacement_quantity <= EPSILON:
                continue
            loss = abs(realized.realized_gain_loss)
            disallowed = loss * min(1.0, replacement_quantity / max(realized.quantity, EPSILON))
            realized.wash_sale_disallowed_loss = float(disallowed)
            realized.taxable_gain_loss = float(realized.realized_gain_loss + disallowed)
            realized.wash_sale_status = self.profile.wash_sale_enforcement

    def remaining_quantity(self, ticker: str) -> float:
        ticker = ticker.upper()
        return float(
            sum(
                lot.remaining_quantity
                for lot in self._lots
                if lot.ticker == ticker and lot.remaining_quantity > EPSILON
            ),
        )

    def open_lots_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for lot in self._lots:
            if lot.remaining_quantity <= EPSILON:
                continue
            rows.append(
                {
                    "lot_id": lot.lot_id,
                    "mode": lot.mode,
                    "account": lot.account,
                    "ticker": lot.ticker,
                    "acquired_at": lot.acquired_at.isoformat(),
                    "quantity": lot.quantity,
                    "remaining_quantity": lot.remaining_quantity,
                    "price": lot.price,
                    "cost_basis_per_share": lot.cost_basis_per_share,
                    "total_cost_basis": lot.total_cost_basis,
                    "source_execution_id": lot.source_execution_id,
                    "fees": lot.fees,
                    "wash_sale_adjustment": lot.wash_sale_adjustment,
                    "current_status": "open",
                }
            )
        return pd.DataFrame(rows)

    def realized_lots_frame(self) -> pd.DataFrame:
        rows = []
        for lot in self._realized:
            rows.append(
                {
                    "realized_id": lot.realized_id,
                    "mode": lot.mode,
                    "account": lot.account,
                    "ticker": lot.ticker,
                    "acquired_at": lot.acquired_at.isoformat(),
                    "sold_at": lot.sold_at.isoformat(),
                    "quantity": lot.quantity,
                    "proceeds": lot.proceeds,
                    "cost_basis": lot.cost_basis,
                    "realized_gain_loss": lot.realized_gain_loss,
                    "wash_sale_disallowed_loss": lot.wash_sale_disallowed_loss,
                    "taxable_gain_loss": lot.taxable_gain_loss,
                    "term": lot.term,
                    "wash_sale_status": lot.wash_sale_status,
                    "source_lot_id": lot.source_lot_id,
                    "source_execution_id": lot.source_execution_id,
                    "sell_execution_id": lot.sell_execution_id,
                }
            )
        return pd.DataFrame(rows)

    def _add_lot(
        self,
        *,
        execution_id: str,
        mode: str,
        account: str,
        ticker: str,
        quantity: float,
        price: float,
        acquired_at: pd.Timestamp,
        fees: float,
    ) -> None:
        total_cost = quantity * price + fees
        lot = TaxLot(
            lot_id=_stable_id(
                "lot", execution_id, ticker, acquired_at.isoformat(), quantity, price
            ),
            mode=mode,
            account=account,
            ticker=ticker,
            acquired_at=acquired_at,
            quantity=float(quantity),
            remaining_quantity=float(quantity),
            price=float(price),
            cost_basis_per_share=float(total_cost / quantity),
            source_execution_id=execution_id,
            fees=float(fees),
        )
        self._lots.append(lot)

    def _sell_lots(
        self,
        *,
        execution_id: str,
        mode: str,
        account: str,
        ticker: str,
        quantity: float,
        price: float,
        sold_at: pd.Timestamp,
        fees: float,
    ) -> list[RealizedTaxLot]:
        remaining = float(quantity)
        realized_rows: list[RealizedTaxLot] = []
        for lot in self._ordered_lots(ticker, price):
            if remaining <= EPSILON:
                break
            consume = min(lot.remaining_quantity, remaining)
            if consume <= EPSILON:
                continue
            fee_alloc = fees * (consume / quantity) if quantity > EPSILON else 0.0
            proceeds = consume * price - fee_alloc
            cost_basis = consume * lot.cost_basis_per_share
            gain_loss = proceeds - cost_basis
            term = (
                "long"
                if (sold_at - lot.acquired_at).days > self.profile.long_term_holding_period_days
                else "short"
            )
            lot.remaining_quantity = max(0.0, lot.remaining_quantity - consume)
            realized = RealizedTaxLot(
                realized_id=_stable_id("realized", execution_id, lot.lot_id, consume, price),
                mode=mode,
                account=account,
                ticker=ticker,
                acquired_at=lot.acquired_at,
                sold_at=sold_at,
                quantity=float(consume),
                proceeds=float(proceeds),
                cost_basis=float(cost_basis),
                realized_gain_loss=float(gain_loss),
                term=term,
                source_lot_id=lot.lot_id,
                source_execution_id=lot.source_execution_id,
                sell_execution_id=execution_id,
            )
            self._realized.append(realized)
            realized_rows.append(realized)
            remaining -= consume
        if remaining > max(EPSILON, quantity * 1e-6):
            raise ValueError(
                f"Cannot sell {quantity} {ticker}; only {quantity - remaining} available."
            )
        return realized_rows

    def _ordered_lots(self, ticker: str, price: float) -> list[TaxLot]:
        lots = [
            lot
            for lot in self._lots
            if lot.ticker == ticker.upper() and lot.remaining_quantity > EPSILON
        ]
        method = self.profile.lot_selection_method
        if method == "fifo":
            return sorted(lots, key=lambda lot: lot.acquired_at)
        if method in {"specific_id_tax_min", "highest_cost"}:
            return sorted(lots, key=lambda lot: (-lot.cost_basis_per_share, lot.acquired_at))
        if method == "lowest_gain":
            return sorted(lots, key=lambda lot: (price - lot.cost_basis_per_share, lot.acquired_at))
        return sorted(lots, key=lambda lot: lot.acquired_at)


def _similar_tickers(ticker: str, substitute_map: dict[str, list[str]] | None) -> set[str]:
    ticker = ticker.upper()
    similar = {ticker}
    for key, values in (substitute_map or {}).items():
        normalized_key = key.upper()
        normalized_values = {value.upper() for value in values}
        if ticker == normalized_key or ticker in normalized_values:
            similar.add(normalized_key)
            similar.update(normalized_values)
    return similar


def _timestamp(value: str | pd.Timestamp | object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def _stable_id(*parts: object) -> str:
    payload = ":".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
