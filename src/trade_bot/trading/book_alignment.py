from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trade_bot.DEFAULTS import DEFAULT_BOOK_ALIGNMENT_MIN_TRADE_WEIGHT
from trade_bot.research.trade_decision import TradeDecisionRun
from trade_bot.trading.journal import TradeJournal


@dataclass(frozen=True)
class BookAlignmentRun:
    summary: pd.DataFrame
    position_plan: pd.DataFrame
    holdings: pd.DataFrame

    @property
    def is_aligned(self) -> bool:
        if self.summary.empty:
            return False
        return str(self.summary.iloc[0].get("alignment_status", "")) == "aligned"


def build_book_alignment(
    *,
    journal: TradeJournal,
    trade_decision: TradeDecisionRun,
    prices: pd.DataFrame,
    mode: str,
    account: str,
    strategy_name: str,
    account_value: float,
    min_trade_weight: float = DEFAULT_BOOK_ALIGNMENT_MIN_TRADE_WEIGHT,
) -> BookAlignmentRun:
    """Reconcile logged account executions to the latest scenario-adjusted target.

    The trade-decision target answers "what should the model want now?". This
    layer answers "what should this paper/live book do from what it already
    holds?" using the local journal as the current book of record.
    """
    if account_value <= 0:
        raise ValueError("Account value must be positive.")
    if min_trade_weight < 0:
        raise ValueError("Minimum trade weight cannot be negative.")

    reference_prices = _latest_prices(prices)
    holdings = _execution_holdings(
        journal=journal,
        mode=mode,
        account=account,
        reference_prices=reference_prices,
    )
    account_value_input = float(account_value)
    account_value, account_value_source, account_value_warning = _effective_account_value(
        account_value_input,
        holdings,
    )
    target_weights = _target_weights(trade_decision.position_plan)
    current_notional = {
        str(row["ticker"]).upper(): float(row["current_notional"])
        for _, row in holdings.iterrows()
    }
    current_quantities = {
        str(row["ticker"]).upper(): float(row["net_quantity"])
        for _, row in holdings.iterrows()
    }
    current_cash_deployed = {
        str(row["ticker"]).upper(): float(row["net_cash_deployed"])
        for _, row in holdings.iterrows()
    }

    tickers = sorted(set(target_weights) | set(current_notional))
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        current_value = float(current_notional.get(ticker, 0.0))
        current_weight = current_value / account_value
        target_weight = float(target_weights.get(ticker, 0.0))
        delta_weight = target_weight - current_weight
        action = _action_from_delta(delta_weight, min_trade_weight=min_trade_weight)
        rows.append(
            {
                "ticker": ticker,
                "current_weight": current_weight,
                "scenario_adjusted_weight": target_weight,
                "target_weight": target_weight,
                "delta_weight": delta_weight,
                "action": action,
                "reference_price": reference_prices.get(ticker),
                "net_quantity": current_quantities.get(ticker, 0.0),
                "net_cash_deployed": current_cash_deployed.get(ticker, 0.0),
                "current_notional": current_value,
                "target_notional": target_weight * account_value,
                "delta_notional": delta_weight * account_value,
            }
        )

    position_plan = pd.DataFrame(rows)
    if not position_plan.empty:
        material = (
            position_plan[["current_weight", "scenario_adjusted_weight", "delta_weight"]]
            .abs()
            .max(axis=1)
            >= 0.0005
        )
        position_plan = position_plan[material].sort_values("delta_weight").reset_index(drop=True)
    else:
        position_plan = _empty_position_plan()

    summary = _alignment_summary(
        position_plan=position_plan,
        holdings=holdings,
        mode=mode,
        account=account,
        strategy_name=strategy_name,
        account_value=account_value,
        account_value_input=account_value_input,
        account_value_source=account_value_source,
        account_value_warning=account_value_warning,
        min_trade_weight=min_trade_weight,
    )
    return BookAlignmentRun(summary=summary, position_plan=position_plan, holdings=holdings)


def latest_book_account_value(
    journal: TradeJournal,
    *,
    mode: str,
    account: str,
    strategy_name: str,
    default: float = 10_000.0,
) -> float:
    snapshots = journal.load_decision_snapshots(limit=1000)
    if snapshots.empty:
        return default
    scoped = snapshots[
        (snapshots["mode"].astype(str) == mode)
        & (snapshots["account"].astype(str) == account)
        & (snapshots["strategy_name"].astype(str) == strategy_name)
    ]
    if scoped.empty:
        return default
    value = pd.to_numeric(scoped.iloc[0].get("account_value"), errors="coerce")
    if pd.isna(value) or float(value) <= 0:
        return default
    return float(value)


def _execution_holdings(
    *,
    journal: TradeJournal,
    mode: str,
    account: str,
    reference_prices: dict[str, float],
) -> pd.DataFrame:
    executions = journal.load_executions(limit=10000, mode=mode, account=account)
    if executions.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "account",
                "ticker",
                "net_quantity",
                "net_cash_deployed",
                "fees",
                "executions",
                "latest_execution",
                "reference_price",
                "current_notional",
            ]
        )

    signed_quantity = executions["quantity"].astype(float).where(
        executions["side"].astype(str).str.upper() == "BUY",
        -executions["quantity"].astype(float),
    )
    signed_notional = executions["notional"].astype(float).where(
        executions["side"].astype(str).str.upper() == "BUY",
        -executions["notional"].astype(float),
    )
    frame = executions.assign(
        ticker=executions["ticker"].astype(str).str.upper(),
        signed_quantity=signed_quantity,
        signed_notional=signed_notional,
    )
    holdings = (
        frame.groupby(["mode", "account", "ticker"], as_index=False)
        .agg(
            net_quantity=("signed_quantity", "sum"),
            net_cash_deployed=("signed_notional", "sum"),
            fees=("fees", "sum"),
            executions=("execution_id", "count"),
            latest_execution=("executed_at_utc", "max"),
        )
        .sort_values(["mode", "account", "ticker"])
    )
    if holdings.empty:
        return holdings
    holdings["reference_price"] = holdings["ticker"].map(reference_prices)
    holdings["current_notional"] = holdings.apply(
        lambda row: _current_notional(row, reference_prices),
        axis=1,
    )
    return holdings


def _target_weights(position_plan: pd.DataFrame) -> dict[str, float]:
    if position_plan.empty:
        return {}
    target_column = (
        "scenario_adjusted_weight"
        if "scenario_adjusted_weight" in position_plan
        else "target_weight"
    )
    if target_column not in position_plan:
        return {}
    targets: dict[str, float] = {}
    for _, row in position_plan.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        value = pd.to_numeric(row.get(target_column), errors="coerce")
        if pd.isna(value):
            continue
        targets[ticker] = float(value)
    return targets


def _alignment_summary(
    *,
    position_plan: pd.DataFrame,
    holdings: pd.DataFrame,
    mode: str,
    account: str,
    strategy_name: str,
    account_value: float,
    account_value_input: float,
    account_value_source: str,
    account_value_warning: str,
    min_trade_weight: float,
) -> pd.DataFrame:
    material = position_plan[position_plan["delta_weight"].abs() >= min_trade_weight].copy()
    max_abs_delta = float(position_plan["delta_weight"].abs().max()) if not position_plan.empty else 0.0
    largest_row = (
        position_plan.loc[position_plan["delta_weight"].abs().idxmax()]
        if not position_plan.empty and max_abs_delta > 0
        else None
    )
    material_trade_count = int(len(material))
    current_weights = _weight_series(position_plan, "current_weight")
    target_weights = _weight_series(position_plan, "scenario_adjusted_weight")
    current_cash_weight = 1.0 - float(current_weights.sum())
    target_cash_weight = 1.0 - float(target_weights.sum())
    has_executions = not holdings.empty and float(holdings["net_quantity"].abs().sum()) > 1e-10
    alignment_status, recommended_action = _alignment_status(
        has_executions=has_executions,
        material_trade_count=material_trade_count,
        max_abs_delta=max_abs_delta,
    )
    largest_ticker = str(largest_row["ticker"]) if largest_row is not None else ""
    largest_delta_notional = (
        float(largest_row["delta_notional"]) if largest_row is not None else 0.0
    )
    explanation = _alignment_explanation(
        alignment_status=alignment_status,
        current_position=_format_weight_vector(current_weights),
        target_position=_format_weight_vector(target_weights),
        largest_ticker=largest_ticker,
        max_abs_delta=max_abs_delta,
        largest_delta_notional=largest_delta_notional,
        min_trade_weight=min_trade_weight,
    )
    return pd.DataFrame(
        [
            {
                "mode": mode,
                "account": account,
                "strategy_name": strategy_name,
                "book_scope": "account",
                "account_value": account_value,
                "account_value_input": account_value_input,
                "account_value_source": account_value_source,
                "account_value_warning": account_value_warning,
                "alignment_status": alignment_status,
                "recommended_action": recommended_action,
                "current_position": _format_weight_vector(current_weights),
                "target_position": _format_weight_vector(target_weights),
                "current_cash_weight": current_cash_weight,
                "target_cash_weight": target_cash_weight,
                "max_abs_delta": max_abs_delta,
                "material_trade_count": material_trade_count,
                "largest_ticker": largest_ticker,
                "largest_delta_weight": float(largest_row["delta_weight"])
                if largest_row is not None
                else 0.0,
                "largest_delta_notional": largest_delta_notional,
                "min_trade_weight": min_trade_weight,
                "has_executions": bool(has_executions),
                "explanation": explanation,
            }
        ]
    )


def _alignment_status(
    *,
    has_executions: bool,
    material_trade_count: int,
    max_abs_delta: float,
) -> tuple[str, str]:
    if material_trade_count == 0:
        return "aligned", "DO_NOTHING"
    if not has_executions:
        return "unstarted", "START_PAPER_BOOK"
    if max_abs_delta >= 0.20 or material_trade_count >= 3:
        return "critical_rebalance", "REVIEW_REBALANCE"
    if max_abs_delta >= 0.05:
        return "rebalance_needed", "REBALANCE"
    return "small_drift", "SMALL_REBALANCE"


def _alignment_explanation(
    *,
    alignment_status: str,
    current_position: str,
    target_position: str,
    largest_ticker: str,
    max_abs_delta: float,
    largest_delta_notional: float,
    min_trade_weight: float,
) -> str:
    if alignment_status == "aligned":
        return (
            f"The selected book is within {min_trade_weight:.0%} of the latest target weights. "
            "No new paper/live tickets are needed unless you deliberately want to refresh prices or relock the decision."
        )
    if alignment_status == "unstarted":
        return (
            f"No logged executions exist for this book yet. Latest target posture is {target_position}; "
            "locking tickets would create the initial paper/live book."
        )
    direction = "add" if largest_delta_notional > 0 else "reduce"
    return (
        f"Current book is {current_position}; latest target is {target_position}. "
        f"Largest actionable drift is {max_abs_delta:.1%} in {largest_ticker}, which implies a "
        f"{direction} of about ${abs(largest_delta_notional):,.0f} before price and size bands."
    )


def _weight_series(position_plan: pd.DataFrame, column: str) -> pd.Series:
    if position_plan.empty or column not in position_plan:
        return pd.Series(dtype=float)
    series = position_plan.set_index("ticker")[column].astype(float)
    return series[series.abs() >= 0.0005].sort_values(ascending=False)


def _format_weight_vector(weights: pd.Series) -> str:
    if weights.empty:
        return "cash/unallocated"
    pieces = [f"{ticker} {float(weight):.0%}" for ticker, weight in weights.items() if abs(weight) >= 0.005]
    return ", ".join(pieces) if pieces else "cash/unallocated"


def _action_from_delta(delta_weight: float, *, min_trade_weight: float) -> str:
    if abs(delta_weight) < min_trade_weight:
        return "HOLD"
    if delta_weight > 0:
        return "ADD"
    return "REDUCE"


def _current_notional(row: pd.Series, reference_prices: dict[str, float]) -> float:
    ticker = str(row.get("ticker", "")).upper()
    reference_price = reference_prices.get(ticker)
    if reference_price is not None:
        return float(row["net_quantity"]) * reference_price
    return float(row["net_cash_deployed"])


def _effective_account_value(
    account_value: float,
    holdings: pd.DataFrame,
) -> tuple[float, str, str]:
    if holdings.empty or "current_notional" not in holdings:
        return account_value, "input", ""
    marked_long_value = float(holdings["current_notional"].clip(lower=0.0).sum())
    if marked_long_value <= account_value + 1e-6:
        return account_value, "input", ""
    return (
        marked_long_value,
        "marked_holdings_floor",
        (
            "Logged holdings exceed the supplied account value; using marked holdings "
            "as the account-value floor for book-weight math."
        ),
    )


def _latest_prices(prices: pd.DataFrame) -> dict[str, float]:
    if prices.empty:
        return {}
    latest = prices.ffill().iloc[-1]
    values: dict[str, float] = {}
    for ticker, value in latest.items():
        value = pd.to_numeric(value, errors="coerce")
        if pd.notna(value) and float(value) > 0:
            values[str(ticker).upper()] = float(value)
    return values


def _empty_position_plan() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "current_weight",
            "scenario_adjusted_weight",
            "target_weight",
            "delta_weight",
            "action",
            "reference_price",
            "net_quantity",
            "net_cash_deployed",
            "current_notional",
            "target_notional",
            "delta_notional",
        ]
    )
