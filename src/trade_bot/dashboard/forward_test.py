from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from trade_bot.dashboard.book_alignment import _render_book_alignment
from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_trade_frame, _safe_timezone
from trade_bot.DEFAULTS import DEFAULT_FORWARD_TEST_ACCOUNT, DEFAULT_FORWARD_TEST_STRATEGY
from trade_bot.research.baselines import BaselineRun
from trade_bot.trading.book_alignment import build_book_alignment
from trade_bot.trading.journal import (
    TicketSizingConfig,
    TradeJournal,
    build_recommendation_tickets,
)

TICKET_COLUMN_HELP = {
    "created_at_utc": "When this recommendation ticket was locked, in UTC. Use it to match the ticket to the market state that produced it.",
    "ticket_id": "Unique ID for a locked recommendation ticket. It links a recommendation to any later paper or live execution.",
    "execution_id": "Unique ID for a logged paper or live execution.",
    "recommendation_id": "Ticket ID that this execution came from. Blank means the execution was entered manually.",
    "status": "Ticket lifecycle: open needs review, executed has been acted on, skipped was intentionally ignored, expired is no longer current.",
    "mode": "Whether the ticket belongs to paper monitoring or live execution tracking.",
    "account": "Local account label used to separate paper/live books or different monitoring sleeves.",
    "strategy_name": "Strategy or operating-system label that generated the ticket. Use this for champion/challenger tracking.",
    "ticker": "Tradable stock or ETF symbol for the ticket.",
    "side": "Execution direction: BUY adds shares; SELL trims or exits shares. This system does not short.",
    "source_action": "Model action before broker translation: ADD means target weight increased; REDUCE means target weight decreased.",
    "current_weight": "Current model/account weight used when building the ticket.",
    "target_weight": "Scenario- and risk-adjusted target weight after the current decision layer.",
    "delta_weight": "Target weight minus current weight. Positive creates a BUY ticket; negative creates a SELL ticket.",
    "reference_price": "Latest available price when the ticket was created. It is an anchor, not an execution guarantee.",
    "limit_low": "Lower acceptable price bound from the configured price band. Use it as review context before execution.",
    "limit_high": "Upper acceptable price bound from the configured price band. Use it as review context before execution.",
    "target_notional": "Dollar value implied by account value times delta weight. Negative values become SELL tickets.",
    "min_notional": "Lower dollar-size bound after the configured size band.",
    "max_notional": "Upper dollar-size bound after the configured size band.",
    "min_shares": "Smallest suggested share quantity after size-band and whole-share settings.",
    "max_shares": "Largest suggested share quantity after size-band and whole-share settings.",
    "rationale": "Human-readable reason copied from the trade-decision layer at ticket creation time.",
    "executed_at_utc": "Actual logged execution timestamp converted to UTC.",
    "quantity": "Executed share quantity.",
    "price": "Executed or logged fill price.",
    "notional": "Execution dollar value before fees.",
    "fees": "Logged execution fees.",
    "notes": "Optional execution notes entered by the user.",
    "net_quantity": "Net shares accumulated from logged executions.",
    "net_cash_deployed": "Net cash deployed from logged executions after buys and sells.",
    "lot_id": "Derived tax-lot ID rebuilt from local execution history.",
    "rebuilt_at_utc": "When the derived tax-lot table was last rebuilt, in UTC.",
    "acquired_at": "Timestamp when the lot was acquired according to local execution history.",
    "sold_at": "Timestamp when the lot was sold according to local execution history.",
    "remaining_quantity": "Shares still open in the derived lot.",
    "cost_basis_per_share": "Estimated cost basis per share for the derived lot, including allocated fees where available.",
    "total_cost_basis": "Estimated remaining cost basis for the open lot.",
    "proceeds": "Estimated sale proceeds for the realized lot after allocated fees.",
    "cost_basis": "Estimated cost basis consumed by the sale.",
    "realized_gain_loss": "Estimated realized gain or loss before wash-sale disallowance.",
    "taxable_gain_loss": "Estimated taxable gain or loss after wash-sale disallowance.",
    "wash_sale_disallowed_loss": "Estimated loss disallowed by configured wash-sale rules.",
    "wash_sale_status": "Configured wash-sale handling applied to this realized lot.",
    "term": "Estimated holding-period bucket: short or long.",
    "source_lot_id": "Open lot consumed by this realized-lot record.",
    "source_execution_id": "Execution that created the original lot.",
    "sell_execution_id": "Execution that realized this lot.",
}


TICKET_LABEL_REFERENCE = pd.DataFrame(
    [
        {
            "label": "open",
            "meaning": "Locked recommendation that still needs review, execution, skipping, or expiration.",
            "how_to_use": "Review before creating more tickets for the same account/strategy.",
        },
        {
            "label": "executed",
            "meaning": "Ticket has been connected to a logged paper or live execution.",
            "how_to_use": "Use for forward-performance audit trails.",
        },
        {
            "label": "skipped",
            "meaning": "Human reviewed the ticket and intentionally did not act.",
            "how_to_use": "Useful when the model suggested action but discretion overrode it.",
        },
        {
            "label": "expired",
            "meaning": "Ticket is stale and should not be acted on without a fresh decision run.",
            "how_to_use": "Use when the market or recommendation changed before execution.",
        },
        {
            "label": "ADD / REDUCE",
            "meaning": "Model-level target-weight direction before broker-side translation.",
            "how_to_use": "ADD usually maps to BUY; REDUCE maps to SELL without shorting.",
        },
        {
            "label": "BUY / SELL",
            "meaning": "Execution-side instruction for the shares in the ticket.",
            "how_to_use": "SELL means trim an existing long position, not open a short.",
        },
        {
            "label": "price band",
            "meaning": "Allowed review range around the reference price.",
            "how_to_use": "If current price is outside the band, refresh/review before logging execution.",
        },
        {
            "label": "size band",
            "meaning": "Allowed sizing range around target notional.",
            "how_to_use": "Lets a human execute a practical quantity without pretending precision is exact.",
        },
    ]
)


def _render_forward_test_and_journal(
    journal: TradeJournal,
    baseline_run: BaselineRun,
) -> None:
    trade_decision = baseline_run.trade_decision
    st.subheader("Forward Test / Trade Journal")
    st.caption(
        "Lock recommendations, paper-trade them, and log actual executions so forward performance can be audited."
    )
    journal_cols = st.columns(4)
    journal_mode = journal_cols[0].selectbox("Mode", ["paper", "live"])
    journal_account = journal_cols[1].text_input("Account", DEFAULT_FORWARD_TEST_ACCOUNT)
    journal_strategy = journal_cols[2].text_input(
        "Strategy label",
        DEFAULT_FORWARD_TEST_STRATEGY,
    )
    account_value = journal_cols[3].number_input(
        "Account value",
        min_value=1.0,
        value=10000.0,
        step=1000.0,
    )

    sizing_cols = st.columns(4)
    price_band_pct = (
        sizing_cols[0].number_input("Price band %", min_value=0.0, value=0.75, step=0.25) / 100.0
    )
    size_band_pct = (
        sizing_cols[1].number_input("Size band %", min_value=0.0, value=20.0, step=5.0) / 100.0
    )
    min_trade_notional = sizing_cols[2].number_input(
        "Min trade $",
        min_value=0.0,
        value=25.0,
        step=25.0,
    )
    whole_shares = sizing_cols[3].checkbox("Whole shares", value=True)
    sizing = TicketSizingConfig(
        account_value=float(account_value),
        price_band_pct=float(price_band_pct),
        size_band_pct=float(size_band_pct),
        min_trade_notional=float(min_trade_notional),
        whole_shares=bool(whole_shares),
    )
    _render_ticket_label_reference()
    book_alignment = build_book_alignment(
        journal=journal,
        trade_decision=trade_decision,
        prices=baseline_run.prices,
        mode=journal_mode,
        account=journal_account,
        strategy_name=journal_strategy,
        account_value=float(account_value),
    )
    _render_book_alignment(book_alignment, heading="Selected Book Alignment")
    alignment_summary = book_alignment.summary.iloc[0] if not book_alignment.summary.empty else {}
    ticket_preview = build_recommendation_tickets(
        trade_decision,
        baseline_run.prices,
        mode=journal_mode,
        account=journal_account,
        strategy_name=journal_strategy,
        sizing=sizing,
        position_plan=book_alignment.position_plan,
        rationale=str(alignment_summary.get("explanation", "")),
    )
    ticket_columns = [
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
    ]
    if ticket_preview.empty:
        st.write(
            "No executable recommendation tickets from the current decision and sizing inputs."
        )
    else:
        st.caption("Preview of tickets that would be locked from the current trade decision.")
        _render_ticket_table(ticket_preview[ticket_columns])
        if st.button("Lock Current Recommendation Set"):
            decision_id = journal.save_decision_snapshot(
                mode=journal_mode,
                account=journal_account,
                strategy_name=journal_strategy,
                trade_decision=trade_decision,
                sizing=sizing,
                tickets=ticket_preview,
            )
            st.success(f"Locked {len(ticket_preview):,} recommendation tickets: {decision_id}")

    st.caption("Locked recommendations")
    ticket_status = st.selectbox("Ticket status", ["open", "all", "executed", "skipped", "expired"])
    stored_tickets = journal.load_recommendation_tickets(
        status=None if ticket_status == "all" else ticket_status
    )
    stored_ticket_columns = [
        "created_at_utc",
        "ticket_id",
        "status",
        "mode",
        "account",
        "strategy_name",
        "ticker",
        "side",
        "reference_price",
        "limit_low",
        "limit_high",
        "target_notional",
        "min_shares",
        "max_shares",
        "rationale",
    ]
    if stored_tickets.empty:
        st.write("No locked recommendation tickets yet.")
    else:
        available_columns = [column for column in stored_ticket_columns if column in stored_tickets]
        _render_ticket_table(stored_tickets[available_columns])

    st.caption("Execution log")
    open_tickets = journal.load_recommendation_tickets(status="open")
    ticket_options = {"manual": None}
    if not open_tickets.empty:
        ticket_options.update(_ticket_option_map(open_tickets))

    with st.form("execution_log_form"):
        selected_ticket_label = st.selectbox(
            "Recommendation ticket",
            list(ticket_options),
            help="Readable ticket label: ticker, side, share range, status, strategy, and short ticket ID. Pick manual for an execution that did not come from a locked recommendation.",
        )
        selected_ticket_id = ticket_options[selected_ticket_label]
        selected_ticket = None
        if selected_ticket_id is not None:
            selected_ticket_rows = open_tickets[open_tickets["ticket_id"] == selected_ticket_id]
            if not selected_ticket_rows.empty:
                selected_ticket = selected_ticket_rows.iloc[0]

        default_ticker = str(selected_ticket["ticker"]) if selected_ticket is not None else "QQQ"
        default_side = str(selected_ticket["side"]) if selected_ticket is not None else "BUY"
        default_price = (
            float(selected_ticket["reference_price"])
            if selected_ticket is not None
            else float(baseline_run.prices.ffill().iloc[-1].get(default_ticker, 0.0))
        )
        default_quantity = (
            float(selected_ticket["min_shares"]) if selected_ticket is not None else 1.0
        )
        execution_cols = st.columns(5)
        execution_ticker = execution_cols[0].text_input("Ticker", default_ticker).upper()
        execution_side = execution_cols[1].selectbox(
            "Side",
            ["BUY", "SELL"],
            index=0 if default_side == "BUY" else 1,
        )
        execution_quantity = execution_cols[2].number_input(
            "Quantity",
            min_value=0.0001,
            value=max(default_quantity, 0.0001),
            step=1.0 if whole_shares else 0.1,
            format="%.4f",
        )
        execution_price = execution_cols[3].number_input(
            "Price",
            min_value=0.01,
            value=max(default_price, 0.01),
            step=0.01,
            format="%.4f",
        )
        execution_fees = execution_cols[4].number_input(
            "Fees",
            min_value=0.0,
            value=0.0,
            step=0.01,
        )

        time_cols = st.columns(3)
        execution_timezone_name = time_cols[0].text_input("Timezone", "America/Denver")
        execution_timezone = _safe_timezone(execution_timezone_name)
        default_execution_time = datetime.now(execution_timezone).replace(microsecond=0)
        execution_date = time_cols[1].date_input("Execution date", default_execution_time.date())
        execution_time = time_cols[2].time_input("Execution time", default_execution_time.time())
        execution_notes = st.text_area("Execution notes", "")
        submitted_execution = st.form_submit_button("Log Execution")

        if submitted_execution:
            local_execution_time = datetime.combine(execution_date, execution_time).replace(
                tzinfo=execution_timezone
            )
            execution_id = journal.log_execution(
                mode=journal_mode,
                account=journal_account,
                ticker=execution_ticker,
                side=execution_side,
                quantity=float(execution_quantity),
                price=float(execution_price),
                executed_at_utc=local_execution_time.astimezone(UTC)
                .replace(microsecond=0)
                .isoformat(),
                recommendation_id=selected_ticket_id,
                fees=float(execution_fees),
                notes=execution_notes,
            )
            st.success(f"Logged execution: {execution_id}")

    if not open_tickets.empty:
        status_cols = st.columns(2)
        status_ticket_options = _ticket_option_map(open_tickets)
        status_ticket_label = status_cols[0].selectbox(
            "Open ticket to update",
            list(status_ticket_options),
            help="Readable ticket label: ticker, side, share range, status, strategy, and short ticket ID.",
        )
        status_ticket = status_ticket_options[status_ticket_label]
        status_update = status_cols[1].selectbox("New status", ["skipped", "expired", "open"])
        if st.button("Update Ticket Status"):
            journal.update_ticket_status(status_ticket, status_update)
            st.success(f"Updated ticket {status_ticket} to {status_update}.")

    executions = journal.load_executions()
    execution_columns = [
        "executed_at_utc",
        "execution_id",
        "recommendation_id",
        "mode",
        "account",
        "ticker",
        "side",
        "quantity",
        "price",
        "notional",
        "fees",
        "notes",
    ]
    if executions.empty:
        st.write("No executions logged yet.")
    else:
        _render_ticket_table(executions[execution_columns])

    position_summary = journal.execution_position_summary(
        mode=journal_mode,
        account=journal_account,
    )
    if not position_summary.empty:
        st.caption("Execution-derived position summary")
        _render_ticket_table(position_summary)

    _render_tax_lot_panel(journal, mode=journal_mode, account=journal_account)


def _render_tax_lot_panel(journal: TradeJournal, *, mode: str, account: str) -> None:
    with st.expander("Estimated taxable lots", expanded=False):
        st.caption(
            "Derived tax-lot tables rebuilt from local executions. Use this for paper/live audit "
            "support only; reconcile broker-reported lots before tax-sensitive real decisions."
        )
        if st.button("Rebuild Tax Lots", key="rebuild_tax_lots"):
            try:
                rebuilt = journal.rebuild_tax_lots(mode=mode, account=account)
                st.success(
                    "Rebuilt tax lots: "
                    f"{len(rebuilt['open_lots']):,} open lots, "
                    f"{len(rebuilt['realized_lots']):,} realized lots."
                )
            except ValueError as exc:
                st.error(f"Could not rebuild tax lots: {exc}")
        open_lots = journal.load_tax_lots(mode=mode, account=account)
        realized_lots = journal.load_tax_realized_lots(mode=mode, account=account)
        lot_tab, realized_tab = st.tabs(["Open Lots", "Realized Lots"])
        with lot_tab:
            if open_lots.empty:
                st.write("No derived open lots yet. Log executions, then rebuild tax lots.")
            else:
                open_columns = [
                    "acquired_at",
                    "lot_id",
                    "mode",
                    "account",
                    "ticker",
                    "remaining_quantity",
                    "cost_basis_per_share",
                    "total_cost_basis",
                    "source_execution_id",
                    "rebuilt_at_utc",
                ]
                _render_ticket_table(
                    open_lots[[column for column in open_columns if column in open_lots]]
                )
        with realized_tab:
            if realized_lots.empty:
                st.write("No derived realized lots yet.")
            else:
                realized_columns = [
                    "sold_at",
                    "ticker",
                    "quantity",
                    "proceeds",
                    "cost_basis",
                    "realized_gain_loss",
                    "wash_sale_disallowed_loss",
                    "taxable_gain_loss",
                    "term",
                    "wash_sale_status",
                    "sell_execution_id",
                    "source_lot_id",
                ]
                _render_ticket_table(
                    realized_lots[
                        [column for column in realized_columns if column in realized_lots]
                    ]
                )


def _render_ticket_table(frame: pd.DataFrame) -> None:
    _render_metric_dataframe(
        _display_trade_frame(frame),
        use_container_width=True,
        column_help=TICKET_COLUMN_HELP,
    )


def _render_ticket_label_reference() -> None:
    with st.expander("Ticket label guide", expanded=False):
        st.caption(
            "Hover over ticket table headers for field definitions. This reference explains the compact labels used in ticket status and execution controls."
        )
        _render_metric_dataframe(TICKET_LABEL_REFERENCE, hide_index=True)


def _ticket_option_map(tickets: pd.DataFrame) -> dict[str, str]:
    options: dict[str, str] = {}
    for _, row in tickets.iterrows():
        ticket_id = str(row["ticket_id"])
        label = _ticket_option_label(row)
        if label in options:
            label = f"{label} | {ticket_id}"
        options[label] = ticket_id
    return options


def _ticket_option_label(row: pd.Series) -> str:
    ticket_id = str(row.get("ticket_id", ""))
    short_ticket_id = ticket_id[:8] if ticket_id else "no-id"
    ticker = str(row.get("ticker", "?")).upper()
    side = str(row.get("side", "?")).upper()
    status = str(row.get("status", "open"))
    strategy = str(row.get("strategy_name", "strategy"))
    share_range = _ticket_share_range(row)
    return f"{ticker} {side} {share_range} | {status} | {strategy} | {short_ticket_id}"


def _ticket_share_range(row: pd.Series) -> str:
    try:
        min_shares = float(row.get("min_shares", 0.0))
        max_shares = float(row.get("max_shares", 0.0))
    except (TypeError, ValueError):
        return "share range n/a"
    return f"{min_shares:.2f}-{max_shares:.2f} sh"
