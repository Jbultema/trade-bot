from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_JOURNAL_PATH,
    DEFAULT_TICKET_MIN_TRADE_NOTIONAL,
    DEFAULT_TICKET_PRICE_BAND_PCT,
    DEFAULT_TICKET_SIZE_BAND_PCT,
    DEFAULT_TICKET_WHOLE_SHARES,
)
from trade_bot.research.trade_decision import TradeDecisionRun
from trade_bot.tax.account import TaxAccountProfile
from trade_bot.tax.lots import TaxLotLedger


@dataclass(frozen=True)
class TicketSizingConfig:
    account_value: float
    price_band_pct: float = DEFAULT_TICKET_PRICE_BAND_PCT
    size_band_pct: float = DEFAULT_TICKET_SIZE_BAND_PCT
    min_trade_notional: float = DEFAULT_TICKET_MIN_TRADE_NOTIONAL
    whole_shares: bool = DEFAULT_TICKET_WHOLE_SHARES


class TradeJournal:
    def __init__(self, path: str | Path = DEFAULT_JOURNAL_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def save_decision_snapshot(
        self,
        *,
        mode: str,
        account: str,
        strategy_name: str,
        trade_decision: TradeDecisionRun,
        sizing: TicketSizingConfig,
        tickets: pd.DataFrame,
        notes: str = "",
    ) -> str:
        decision_id = str(uuid.uuid4())
        created_at_utc = utc_now_iso()
        summary = _first_row(trade_decision.summary)
        position_plan_json = _frame_json(trade_decision.position_plan)
        evidence_json = _frame_json(trade_decision.evidence)
        scenario_links_json = _frame_json(trade_decision.scenario_links)
        decision_hash = _decision_hash(
            summary=summary,
            position_plan_json=position_plan_json,
            evidence_json=evidence_json,
            scenario_links_json=scenario_links_json,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO decision_snapshots (
                    decision_id,
                    created_at_utc,
                    mode,
                    account,
                    strategy_name,
                    decision_hash,
                    account_value,
                    recommended_action,
                    risk_status,
                    risk_budget_multiplier,
                    base_position,
                    scenario_adjusted_position,
                    human_explanation,
                    position_plan_json,
                    evidence_json,
                    scenario_links_json,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    created_at_utc,
                    mode,
                    account,
                    strategy_name,
                    decision_hash,
                    float(sizing.account_value),
                    str(summary.get("recommended_action", "")),
                    str(summary.get("risk_status", "")),
                    _optional_float(summary.get("risk_budget_multiplier")),
                    str(summary.get("base_position", "")),
                    str(summary.get("scenario_adjusted_position", "")),
                    str(summary.get("human_explanation", "")),
                    position_plan_json,
                    evidence_json,
                    scenario_links_json,
                    notes,
                ),
            )
            self._insert_tickets(connection, decision_id, created_at_utc, tickets)
        return decision_id

    def log_execution(
        self,
        *,
        mode: str,
        account: str,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        executed_at_utc: str,
        recommendation_id: str | None = None,
        fees: float = 0.0,
        notes: str = "",
    ) -> str:
        execution_id = str(uuid.uuid4())
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            msg = f"Unsupported execution side: {side}"
            raise ValueError(msg)
        if quantity <= 0:
            raise ValueError("Execution quantity must be positive.")
        if price <= 0:
            raise ValueError("Execution price must be positive.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO executions (
                    execution_id,
                    recommendation_id,
                    created_at_utc,
                    executed_at_utc,
                    mode,
                    account,
                    ticker,
                    side,
                    quantity,
                    price,
                    notional,
                    fees,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    recommendation_id or "",
                    utc_now_iso(),
                    executed_at_utc,
                    mode,
                    account,
                    ticker.upper(),
                    side,
                    float(quantity),
                    float(price),
                    float(quantity * price),
                    float(fees),
                    notes,
                ),
            )
            if recommendation_id:
                connection.execute(
                    """
                    UPDATE recommendation_tickets
                    SET status = 'executed',
                        updated_at_utc = ?
                    WHERE ticket_id = ?
                    """,
                    (utc_now_iso(), recommendation_id),
                )
        return execution_id

    def update_ticket_status(self, ticket_id: str, status: str) -> None:
        if status not in {"open", "executed", "skipped", "expired"}:
            msg = f"Unsupported ticket status: {status}"
            raise ValueError(msg)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE recommendation_tickets
                SET status = ?,
                    updated_at_utc = ?
                WHERE ticket_id = ?
                """,
                (status, utc_now_iso(), ticket_id),
            )

    def load_decision_snapshots(self, *, limit: int = 50) -> pd.DataFrame:
        return self._read_sql(
            """
            SELECT *
            FROM decision_snapshots
            ORDER BY created_at_utc DESC
            LIMIT ?
            """,
            (limit,),
        )

    def load_recommendation_tickets(
        self,
        *,
        status: str | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        if status:
            return self._read_sql(
                """
                SELECT *
                FROM recommendation_tickets
                WHERE status = ?
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (status, limit),
            )
        return self._read_sql(
            """
            SELECT *
            FROM recommendation_tickets
            ORDER BY created_at_utc DESC
            LIMIT ?
            """,
            (limit,),
        )

    def load_executions(
        self,
        *,
        limit: int = 200,
        mode: str | None = None,
        account: str | None = None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: list[object] = []
        if mode is not None:
            conditions.append("mode = ?")
            params.append(mode)
        if account is not None:
            conditions.append("account = ?")
            params.append(account)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT *
            FROM executions
            {where_clause}
            ORDER BY executed_at_utc DESC, created_at_utc DESC
            LIMIT ?
            """
        params.append(limit)
        return self._read_sql(query, tuple(params))

    def execution_position_summary(
        self,
        *,
        mode: str | None = None,
        account: str | None = None,
    ) -> pd.DataFrame:
        executions = self.load_executions(limit=10000, mode=mode, account=account)
        if executions.empty:
            return pd.DataFrame()
        signed_quantity = (
            executions["quantity"]
            .astype(float)
            .where(
                executions["side"] == "BUY",
                -executions["quantity"].astype(float),
            )
        )
        signed_notional = (
            executions["notional"]
            .astype(float)
            .where(
                executions["side"] == "BUY",
                -executions["notional"].astype(float),
            )
        )
        frame = executions.assign(
            signed_quantity=signed_quantity,
            signed_notional=signed_notional,
        )
        summary = (
            frame.groupby(["mode", "account", "ticker"], as_index=False)
            .agg(
                net_quantity=("signed_quantity", "sum"),
                net_cash_deployed=("signed_notional", "sum"),
                executions=("execution_id", "count"),
                latest_execution=("executed_at_utc", "max"),
            )
            .sort_values(["mode", "account", "ticker"])
        )
        return summary

    def rebuild_tax_lots(
        self,
        *,
        mode: str | None = None,
        account: str | None = None,
        profile: TaxAccountProfile | None = None,
        substitute_map: dict[str, list[str]] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Rebuild derived tax-lot tables from execution history."""

        executions = self.load_executions(limit=100000, mode=mode, account=account)
        ledger = TaxLotLedger(profile or TaxAccountProfile())
        if not executions.empty:
            ledger.process_frame(executions)
            ledger.apply_wash_sale_rules(substitute_map=substitute_map)
        open_lots = ledger.open_lots_frame()
        realized_lots = ledger.realized_lots_frame()
        rebuilt_at_utc = utc_now_iso()
        with self._connect() as connection:
            self._delete_tax_rows(connection, "tax_lots", mode=mode, account=account)
            self._delete_tax_rows(connection, "tax_realized_lots", mode=mode, account=account)
            self._insert_tax_lots(connection, open_lots, rebuilt_at_utc)
            self._insert_tax_realized_lots(connection, realized_lots, rebuilt_at_utc)
        return {"open_lots": open_lots, "realized_lots": realized_lots}

    def load_tax_lots(
        self,
        *,
        limit: int = 500,
        mode: str | None = None,
        account: str | None = None,
    ) -> pd.DataFrame:
        return self._load_tax_table("tax_lots", limit=limit, mode=mode, account=account)

    def load_tax_realized_lots(
        self,
        *,
        limit: int = 500,
        mode: str | None = None,
        account: str | None = None,
    ) -> pd.DataFrame:
        return self._load_tax_table("tax_realized_lots", limit=limit, mode=mode, account=account)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_snapshots (
                    decision_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    account TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    decision_hash TEXT NOT NULL,
                    account_value REAL NOT NULL,
                    recommended_action TEXT NOT NULL,
                    risk_status TEXT NOT NULL,
                    risk_budget_multiplier REAL,
                    base_position TEXT NOT NULL,
                    scenario_adjusted_position TEXT NOT NULL,
                    human_explanation TEXT NOT NULL,
                    position_plan_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    scenario_links_json TEXT NOT NULL,
                    notes TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    account TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    source_action TEXT NOT NULL,
                    current_weight REAL NOT NULL,
                    target_weight REAL NOT NULL,
                    delta_weight REAL NOT NULL,
                    reference_price REAL NOT NULL,
                    limit_low REAL NOT NULL,
                    limit_high REAL NOT NULL,
                    target_notional REAL NOT NULL,
                    min_notional REAL NOT NULL,
                    max_notional REAL NOT NULL,
                    min_shares REAL NOT NULL,
                    max_shares REAL NOT NULL,
                    status TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decision_snapshots(decision_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tax_lots (
                    lot_id TEXT PRIMARY KEY,
                    rebuilt_at_utc TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    account TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    remaining_quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    cost_basis_per_share REAL NOT NULL,
                    total_cost_basis REAL NOT NULL,
                    source_execution_id TEXT NOT NULL,
                    fees REAL NOT NULL,
                    wash_sale_adjustment REAL NOT NULL,
                    current_status TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tax_realized_lots (
                    realized_id TEXT PRIMARY KEY,
                    rebuilt_at_utc TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    account TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    sold_at TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    proceeds REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    realized_gain_loss REAL NOT NULL,
                    wash_sale_disallowed_loss REAL NOT NULL,
                    taxable_gain_loss REAL NOT NULL,
                    term TEXT NOT NULL,
                    wash_sale_status TEXT NOT NULL,
                    source_lot_id TEXT NOT NULL,
                    source_execution_id TEXT NOT NULL,
                    sell_execution_id TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    recommendation_id TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    executed_at_utc TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    account TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    notional REAL NOT NULL,
                    fees REAL NOT NULL,
                    notes TEXT NOT NULL
                )
                """
            )

    def _load_tax_table(
        self,
        table: str,
        *,
        limit: int,
        mode: str | None,
        account: str | None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: list[object] = []
        if mode is not None:
            conditions.append("mode = ?")
            params.append(mode)
        if account is not None:
            conditions.append("account = ?")
            params.append(account)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_column = "acquired_at" if table == "tax_lots" else "sold_at"
        query = f"""
            SELECT *
            FROM {table}
            {where_clause}
            ORDER BY {order_column} DESC
            LIMIT ?
            """
        params.append(limit)
        return self._read_sql(query, tuple(params))

    def _delete_tax_rows(
        self,
        connection: sqlite3.Connection,
        table: str,
        *,
        mode: str | None,
        account: str | None,
    ) -> None:
        conditions: list[str] = []
        params: list[object] = []
        if mode is not None:
            conditions.append("mode = ?")
            params.append(mode)
        if account is not None:
            conditions.append("account = ?")
            params.append(account)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        connection.execute(f"DELETE FROM {table} {where_clause}", tuple(params))

    def _insert_tax_lots(
        self,
        connection: sqlite3.Connection,
        lots: pd.DataFrame,
        rebuilt_at_utc: str,
    ) -> None:
        columns = [
            "lot_id",
            "rebuilt_at_utc",
            "mode",
            "account",
            "ticker",
            "acquired_at",
            "quantity",
            "remaining_quantity",
            "price",
            "cost_basis_per_share",
            "total_cost_basis",
            "source_execution_id",
            "fees",
            "wash_sale_adjustment",
            "current_status",
        ]
        for _, row in lots.iterrows():
            payload = {column: row.get(column, "") for column in columns}
            payload["rebuilt_at_utc"] = rebuilt_at_utc
            connection.execute(
                """
                INSERT INTO tax_lots (
                    lot_id, rebuilt_at_utc, mode, account, ticker, acquired_at,
                    quantity, remaining_quantity, price, cost_basis_per_share,
                    total_cost_basis, source_execution_id, fees, wash_sale_adjustment,
                    current_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(payload[column] for column in columns),
            )

    def _insert_tax_realized_lots(
        self,
        connection: sqlite3.Connection,
        lots: pd.DataFrame,
        rebuilt_at_utc: str,
    ) -> None:
        columns = [
            "realized_id",
            "rebuilt_at_utc",
            "mode",
            "account",
            "ticker",
            "acquired_at",
            "sold_at",
            "quantity",
            "proceeds",
            "cost_basis",
            "realized_gain_loss",
            "wash_sale_disallowed_loss",
            "taxable_gain_loss",
            "term",
            "wash_sale_status",
            "source_lot_id",
            "source_execution_id",
            "sell_execution_id",
        ]
        for _, row in lots.iterrows():
            payload = {column: row.get(column, "") for column in columns}
            payload["rebuilt_at_utc"] = rebuilt_at_utc
            connection.execute(
                """
                INSERT INTO tax_realized_lots (
                    realized_id, rebuilt_at_utc, mode, account, ticker, acquired_at,
                    sold_at, quantity, proceeds, cost_basis, realized_gain_loss,
                    wash_sale_disallowed_loss, taxable_gain_loss, term, wash_sale_status,
                    source_lot_id, source_execution_id, sell_execution_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(payload[column] for column in columns),
            )

    def _insert_tickets(
        self,
        connection: sqlite3.Connection,
        decision_id: str,
        created_at_utc: str,
        tickets: pd.DataFrame,
    ) -> None:
        for _, row in tickets.iterrows():
            connection.execute(
                """
                INSERT INTO recommendation_tickets (
                    ticket_id,
                    decision_id,
                    created_at_utc,
                    updated_at_utc,
                    mode,
                    account,
                    strategy_name,
                    ticker,
                    side,
                    source_action,
                    current_weight,
                    target_weight,
                    delta_weight,
                    reference_price,
                    limit_low,
                    limit_high,
                    target_notional,
                    min_notional,
                    max_notional,
                    min_shares,
                    max_shares,
                    status,
                    rationale
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["ticket_id"]),
                    decision_id,
                    created_at_utc,
                    created_at_utc,
                    str(row["mode"]),
                    str(row["account"]),
                    str(row["strategy_name"]),
                    str(row["ticker"]),
                    str(row["side"]),
                    str(row["source_action"]),
                    float(row["current_weight"]),
                    float(row["target_weight"]),
                    float(row["delta_weight"]),
                    float(row["reference_price"]),
                    float(row["limit_low"]),
                    float(row["limit_high"]),
                    float(row["target_notional"]),
                    float(row["min_notional"]),
                    float(row["max_notional"]),
                    float(row["min_shares"]),
                    float(row["max_shares"]),
                    str(row["status"]),
                    str(row["rationale"]),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _read_sql(self, query: str, params: tuple[object, ...]) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(query, connection, params=params)


def build_recommendation_tickets(
    trade_decision: TradeDecisionRun,
    prices: pd.DataFrame,
    *,
    mode: str,
    account: str,
    strategy_name: str,
    sizing: TicketSizingConfig,
    position_plan: pd.DataFrame | None = None,
    rationale: str | None = None,
) -> pd.DataFrame:
    plan = trade_decision.position_plan if position_plan is None else position_plan
    if plan.empty:
        return _empty_ticket_frame()
    if sizing.account_value <= 0:
        raise ValueError("Account value must be positive.")
    if sizing.price_band_pct < 0:
        raise ValueError("Price band cannot be negative.")
    if sizing.size_band_pct < 0:
        raise ValueError("Size band cannot be negative.")

    reference_prices = _latest_prices(prices)
    summary = _first_row(trade_decision.summary)
    ticket_rationale = str(
        rationale if rationale is not None else summary.get("human_explanation", "")
    )
    rows: list[dict[str, object]] = []
    for _, row in plan.iterrows():
        delta_weight = float(row["delta_weight"])
        if abs(delta_weight) < 1e-8:
            continue
        source_action = str(row.get("action", "")).upper()
        if source_action not in {"ADD", "REDUCE"}:
            continue
        ticker = str(row["ticker"]).upper()
        if ticker not in reference_prices:
            continue
        reference_price = reference_prices[ticker]
        target_notional = delta_weight * sizing.account_value
        if abs(target_notional) < sizing.min_trade_notional:
            continue
        side = "BUY" if target_notional > 0 else "SELL"
        abs_target = abs(target_notional)
        min_notional = abs_target * max(0.0, 1.0 - sizing.size_band_pct)
        max_notional = abs_target * (1.0 + sizing.size_band_pct)
        min_shares = _shares_for_notional(
            min_notional,
            reference_price,
            whole_shares=sizing.whole_shares,
        )
        max_shares = _shares_for_notional(
            max_notional,
            reference_price,
            whole_shares=sizing.whole_shares,
        )
        if max_shares <= 0:
            continue
        rows.append(
            {
                "ticket_id": str(uuid.uuid4()),
                "mode": mode,
                "account": account,
                "strategy_name": strategy_name,
                "ticker": ticker,
                "side": side,
                "source_action": source_action,
                "current_weight": float(row["current_weight"]),
                "target_weight": float(row["scenario_adjusted_weight"]),
                "delta_weight": delta_weight,
                "reference_price": reference_price,
                "limit_low": reference_price * (1.0 - sizing.price_band_pct),
                "limit_high": reference_price * (1.0 + sizing.price_band_pct),
                "target_notional": target_notional,
                "min_notional": min_notional,
                "max_notional": max_notional,
                "min_shares": min_shares,
                "max_shares": max_shares,
                "status": "open",
                "rationale": ticket_rationale,
            }
        )

    if not rows:
        return _empty_ticket_frame()
    return pd.DataFrame(rows)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _latest_prices(prices: pd.DataFrame) -> dict[str, float]:
    latest = prices.ffill().iloc[-1]
    values: dict[str, float] = {}
    for ticker, value in latest.items():
        numeric = _optional_float(value)
        if numeric is not None and numeric > 0:
            values[str(ticker).upper()] = numeric
    return values


def _shares_for_notional(notional: float, price: float, *, whole_shares: bool) -> float:
    raw_shares = notional / price
    if whole_shares:
        return float(int(raw_shares))
    return round(raw_shares, 4)


def _first_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _frame_json(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "[]"
    return frame.to_json(orient="records", date_format="iso")


def _decision_hash(
    *,
    summary: dict[str, Any],
    position_plan_json: str,
    evidence_json: str,
    scenario_links_json: str,
) -> str:
    payload = json.dumps(
        {
            "summary": summary,
            "position_plan": json.loads(position_plan_json),
            "evidence": json.loads(evidence_json),
            "scenario_links": json.loads(scenario_links_json),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _empty_ticket_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticket_id",
            "mode",
            "account",
            "strategy_name",
            "ticker",
            "side",
            "source_action",
            "current_weight",
            "target_weight",
            "delta_weight",
            "reference_price",
            "limit_low",
            "limit_high",
            "target_notional",
            "min_notional",
            "max_notional",
            "min_shares",
            "max_shares",
            "status",
            "rationale",
        ]
    )
