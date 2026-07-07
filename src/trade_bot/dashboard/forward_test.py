from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trade_bot.dashboard.book_alignment import _render_book_alignment
from trade_bot.dashboard.components import _clearable_selectbox, _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_trade_frame, _safe_timezone
from trade_bot.dashboard.ticket_explainers import ticket_column_help
from trade_bot.DEFAULTS import DEFAULT_FORWARD_TEST_ACCOUNT, DEFAULT_FORWARD_TEST_STRATEGY
from trade_bot.research.approach_explorer import (
    build_approach_backtest_result,
    decision_sanity_from_catalog_row,
    execution_for_catalog_row,
    future_state_model_from_catalog_row,
    scenario_sizing_from_catalog_row,
    strategy_drawdown_model_from_catalog_row,
    strategy_from_catalog_row,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.storage.warehouse import TradingWarehouse
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
TICKET_COLUMN_HELP = {**TICKET_COLUMN_HELP, **ticket_column_help()}


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
    *,
    bot_config: object | None = None,
    warehouse_path: str | Path | None = None,
) -> None:
    trade_decision = baseline_run.trade_decision
    st.subheader("Forward Test / Trade Journal")
    st.caption(
        "Operational record keeping for paper/live decisions: configure the book, review alignment, lock recommendation tickets, log executions, and audit what happened."
    )

    st.markdown(
        """
        <div class="brief-grid">
            <div class="brief-card brief-card-warning">
                <p class="brief-label">Step 1</p>
                <p class="brief-answer">Review book alignment</p>
                <p class="brief-detail">Check whether the selected paper/live book is already close enough to the latest target posture.</p>
            </div>
            <div class="brief-card brief-card-warning">
                <p class="brief-label">Step 2</p>
                <p class="brief-answer">Lock tickets only when needed</p>
                <p class="brief-detail">Create a dated recommendation set when the current decision should become an auditable paper/live action.</p>
            </div>
            <div class="brief-card brief-card-success">
                <p class="brief-label">Step 3</p>
                <p class="brief-answer">Log fills after execution</p>
                <p class="brief-detail">Record exact ticker, side, quantity, price, time, fees, and notes so performance can be reconciled.</p>
            </div>
            <div class="brief-card">
                <p class="brief-label">Step 4</p>
                <p class="brief-answer">Audit records</p>
                <p class="brief-detail">Use ledgers, position summary, tax-lot estimates, and allocation history to verify the forward book.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Record setup")
    st.caption(
        "These inputs define which local book receives tickets and executions. They do not change the strategy backtest."
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

    with st.expander("Ticket sizing controls", expanded=False):
        st.caption(
            "Advanced controls for translating target-weight changes into practical ticket sizes."
        )
        sizing_cols = st.columns(4)
        price_band_pct = (
            sizing_cols[0].number_input("Price band %", min_value=0.0, value=0.75, step=0.25)
            / 100.0
        )
        size_band_pct = (
            sizing_cols[1].number_input("Size band %", min_value=0.0, value=20.0, step=5.0)
            / 100.0
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

    st.markdown("### 1. Book alignment")
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

    st.markdown("### 2. Recommendation tickets")
    st.caption(
        "Tickets are the bridge between the model recommendation and a human-reviewed paper/live action. Lock only when you want the current recommendation set preserved for audit."
    )
    _render_ticket_label_reference()
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
    with st.expander("Current recommendation ticket preview", expanded=not ticket_preview.empty):
        if ticket_preview.empty:
            st.write(
                "No executable recommendation tickets from the current decision and sizing inputs."
            )
        else:
            st.caption("These are not locked yet. Review them before creating an auditable ticket set.")
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

    open_tickets = journal.load_recommendation_tickets(status="open")
    ticket_status = st.selectbox(
        "Ticket ledger filter",
        ["open", "all", "executed", "skipped", "expired"],
        help="Filter the locked recommendation ledger. Open tickets still need execution, skip, or expiration.",
    )
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
    record_cols = st.columns(3)
    record_cols[0].metric("Open Tickets", len(open_tickets))
    record_cols[1].metric("Filtered Tickets", len(stored_tickets))
    record_cols[2].metric("Filter", ticket_status.title())
    with st.expander("Locked recommendation ledger", expanded=False):
        if stored_tickets.empty:
            st.write("No locked recommendation tickets for this filter.")
        else:
            available_columns = [column for column in stored_ticket_columns if column in stored_tickets]
            _render_ticket_table(stored_tickets[available_columns])

    st.markdown("### 3. Execution journal")
    st.caption(
        "After you paper-trade or live-trade a ticket, log the exact fill here. Manual entries are allowed for executions that did not come from a locked ticket."
    )
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
        with st.expander("Update open ticket status", expanded=False):
            status_cols = st.columns(2)
            status_ticket_options = _ticket_option_map(open_tickets)
            with status_cols[0]:
                status_ticket_label = _clearable_selectbox(
                    "Open ticket to update",
                    list(status_ticket_options),
                    key="status_ticket_label",
                    help=(
                        "Readable ticket label: ticker, side, share range, status, strategy, "
                        "and short ticket ID."
                    ),
                    placeholder="Search open tickets...",
                )
            if status_ticket_label is None:
                st.info("Choose an open ticket before updating status.")
            else:
                status_ticket = status_ticket_options[status_ticket_label]
                status_update = status_cols[1].selectbox(
                    "New status", ["skipped", "expired", "open"]
                )
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
    with st.expander("Execution ledger", expanded=False):
        if executions.empty:
            st.write("No executions logged yet.")
        else:
            _render_ticket_table(executions[execution_columns])

    st.markdown("### 4. Records and audit")
    st.caption(
        "Use these derived records to reconcile the book after trades have been logged."
    )
    position_summary = journal.execution_position_summary(
        mode=journal_mode,
        account=journal_account,
    )
    with st.expander("Execution-derived position summary", expanded=not position_summary.empty):
        if position_summary.empty:
            st.write("No execution-derived positions yet for the selected mode/account.")
        else:
            _render_ticket_table(position_summary)

    _render_tax_lot_panel(journal, mode=journal_mode, account=journal_account)
    _render_forward_allocation_history(
        baseline_run=baseline_run,
        bot_config=bot_config,
        warehouse_path=warehouse_path,
        mode=journal_mode,
        account=journal_account,
    )


def _render_forward_allocation_history(
    *,
    baseline_run: BaselineRun,
    bot_config: object | None,
    warehouse_path: str | Path | None,
    mode: str,
    account: str,
) -> None:
    st.subheader("Forward Allocation History")
    st.caption(
        "Compare the strategy's historical target-weight path with the allocation record "
        "captured after a paper/live monitoring window started."
    )
    if warehouse_path is None:
        st.info("No warehouse path is configured, so forward allocation records are unavailable.")
        return

    try:
        warehouse = TradingWarehouse(Path(warehouse_path))
        windows = warehouse.list_monitoring_windows(status="active")
        valuations = warehouse.read_table("strategy_daily_valuations")
    except Exception as exc:  # pragma: no cover - defensive dashboard guard
        st.warning(f"Could not load monitoring allocation history: {exc}")
        return

    if windows.empty:
        st.info("No active monitoring windows are configured yet.")
        return
    if valuations.empty or "latest_weights_json" not in valuations:
        st.info(
            "No forward allocation valuation rows are available yet. Run paper valuation "
            "after seeding monitoring windows to populate this chart."
        )
        return

    all_window_options = _forward_window_options(windows, valuations)
    if all_window_options.empty:
        st.info("No valued monitoring windows have parseable allocation records yet.")
        return

    scoped_window_options = _forward_window_options(
        windows,
        valuations,
        mode=mode,
        account=account,
    )
    scope_options = ["All valued monitoring windows"]
    if not scoped_window_options.empty:
        scope_options.append(f"Selected book only ({mode}/{account})")
    selected_scope = st.selectbox(
        "Forward allocation source",
        scope_options,
        help=(
            "Use all valued monitoring windows to compare champion/challenger/reference "
            "strategies. Use selected book only when auditing the exact mode/account chosen above."
        ),
        key="forward_allocation_scope",
    )
    window_options = (
        scoped_window_options
        if selected_scope.startswith("Selected book only")
        else all_window_options
    )
    if window_options.empty:
        st.info("No valued monitoring windows match the selected allocation source.")
        return

    selected_label = _clearable_selectbox(
        "Forward allocation strategy",
        window_options["label"].tolist(),
        key="forward_allocation_strategy",
        placeholder="Search monitored strategies...",
    )
    if selected_label is None:
        st.info("Choose a monitored strategy to inspect its allocation history.")
        return
    selected_window = window_options[window_options["label"] == selected_label].iloc[0]
    strategy_name = str(selected_window["strategy_name"])
    window_id = str(selected_window["window_id"])
    start_date = pd.to_datetime(selected_window.get("start_date"), errors="coerce")

    historical_weights, historical_note = _historical_weight_history_for_strategy(
        strategy_name,
        baseline_run=baseline_run,
        warehouse=warehouse,
        bot_config=bot_config,
    )
    forward_weights = _forward_weight_history_from_valuations(valuations, window_id=window_id)

    if forward_weights.empty:
        st.info("This monitoring window has no parsed forward allocation records yet.")
        return

    view = st.selectbox(
        "Allocation history window",
        [
            "Full history",
            "5Y before start + forward",
            "1Y before start + forward",
            "Forward overlap",
            "Forward records only",
            "Custom",
        ],
        key="forward_allocation_history_window",
    )
    historical_weights, forward_weights = _apply_allocation_view_window(
        historical_weights,
        forward_weights,
        view=view,
        start_date=start_date,
    )
    if view == "Custom":
        min_date, max_date = _allocation_date_bounds(historical_weights, forward_weights)
        if min_date is not None and max_date is not None:
            min_allowed = min_date.date()
            max_allowed = max_date.date()
            start_key = "forward_allocation_custom_start"
            end_key = "forward_allocation_custom_end"
            st.session_state[start_key] = _clamp_date_value(
                st.session_state.get(start_key),
                min_value=min_allowed,
                max_value=max_allowed,
                fallback=min_allowed,
            )
            st.session_state[end_key] = _clamp_date_value(
                st.session_state.get(end_key),
                min_value=min_allowed,
                max_value=max_allowed,
                fallback=max_allowed,
            )
            date_cols = st.columns(2)
            custom_start = date_cols[0].date_input(
                "Allocation start",
                st.session_state[start_key],
                min_value=min_allowed,
                max_value=max_allowed,
                key=start_key,
            )
            custom_end = date_cols[1].date_input(
                "Allocation end",
                st.session_state[end_key],
                min_value=min_allowed,
                max_value=max_allowed,
                key=end_key,
            )
            if custom_start > custom_end:
                st.warning("Allocation start is after allocation end; showing the full available window.")
                custom_start = min_allowed
                custom_end = max_allowed
            historical_weights = _filter_weight_history(
                historical_weights,
                start=custom_start,
                end=custom_end,
            )
            forward_weights = _filter_weight_history(
                forward_weights,
                start=custom_start,
                end=custom_end,
            )

    historical_weights, forward_weights = _prepare_allocation_frames(
        historical_weights,
        forward_weights,
        max_assets=10,
    )
    if historical_weights.empty and forward_weights.empty:
        st.info("No allocation rows remain for the selected window.")
        return

    stats_cols = st.columns(4)
    latest_forward = forward_weights.iloc[-1] if not forward_weights.empty else pd.Series()
    latest_label = forward_weights.index.max().date() if not forward_weights.empty else "n/a"
    stats_cols[0].metric("Forward Start", _format_date_for_metric(start_date))
    stats_cols[1].metric("Latest Forward Weight Date", str(latest_label))
    stats_cols[2].metric("Forward Records", f"{len(forward_weights):,}")
    stats_cols[3].metric(
        "Latest Risk Assets",
        _format_percent_value(1.0 - float(latest_forward.get("cash_or_unallocated", 0.0))),
    )

    figure = _make_forward_allocation_history_figure(
        historical_weights,
        forward_weights,
        start_date=start_date,
        strategy_name=strategy_name,
    )
    st.plotly_chart(figure, use_container_width=True)

    if historical_note:
        st.caption(historical_note)

    table_cols = st.columns(2)
    with table_cols[0]:
        st.caption("Latest forward allocation")
        _render_metric_dataframe(
            _display_trade_frame(_latest_weight_table(forward_weights)),
            hide_index=True,
            use_container_width=True,
        )
    with table_cols[1]:
        st.caption("Monitoring window")
        _render_metric_dataframe(
            pd.DataFrame(
                [
                    {
                        "field": "window_role",
                        "value": selected_window.get("window_role", ""),
                    },
                    {"field": "mode", "value": selected_window.get("mode", "")},
                    {"field": "account", "value": selected_window.get("account", "")},
                    {"field": "benchmark", "value": selected_window.get("benchmark", "")},
                    {
                        "field": "start_date",
                        "value": selected_window.get("start_date", ""),
                    },
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


def _forward_window_options(
    windows: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    mode: str | None = None,
    account: str | None = None,
) -> pd.DataFrame:
    valued = valuations.copy()
    if "latest_weights_json" not in valued:
        return pd.DataFrame()
    valued = valued[
        valued["latest_weights_json"].apply(lambda raw: bool(_parse_weights_json(raw)))
    ].copy()
    if valued.empty:
        return pd.DataFrame()
    valued["window_id"] = valued["window_id"].astype(str)
    valuation_counts = (
        valued.groupby("window_id")
        .agg(
            valuation_rows=("valuation_date", "count"),
            latest_valuation_date=("valuation_date", "max"),
        )
        .reset_index()
    )
    options = windows.copy()
    options["window_id"] = options["window_id"].astype(str)
    options = options.merge(valuation_counts, on="window_id", how="inner")
    if mode is not None and account is not None:
        scoped = options[
            (options["mode"].astype(str) == str(mode))
            & (options["account"].astype(str) == str(account))
        ]
    else:
        scoped = options
    if scoped.empty:
        return scoped
    scoped = scoped.sort_values(
        ["window_role", "strategy_name", "latest_valuation_date"],
        ascending=[True, True, False],
    ).copy()
    scoped["label"] = scoped.apply(_forward_window_label, axis=1)
    return scoped


def _forward_window_label(row: pd.Series) -> str:
    role = str(row.get("window_role", "window"))
    strategy = str(row.get("strategy_name", "strategy"))
    scope = f"{row.get('mode', '')}/{row.get('account', '')}"
    start = str(row.get("start_date", ""))
    latest = str(row.get("latest_valuation_date", ""))
    count = int(row.get("valuation_rows", 0) or 0)
    return f"{role} | {strategy} | {scope} | start {start} | {count} records thru {latest}"


def _historical_weight_history_for_strategy(
    strategy_name: str,
    *,
    baseline_run: BaselineRun,
    warehouse: TradingWarehouse,
    bot_config: object | None,
) -> tuple[pd.DataFrame, str]:
    if strategy_name in baseline_run.results:
        result = baseline_run.results[strategy_name]
        return _normalize_weight_history(result.weights), ""

    reconstructed = _reconstruct_candidate_weight_history(
        strategy_name,
        baseline_run=baseline_run,
        warehouse=warehouse,
        bot_config=bot_config,
    )
    if not reconstructed.empty:
        return (
            reconstructed,
            "Historical allocation history was reconstructed from the stored experiment "
            "manifest for this monitored candidate.",
        )
    return (
        pd.DataFrame(),
        "Historical allocation history is not available in the current snapshot for this "
        "strategy, so the chart shows only the forward-tested allocation record.",
    )


def _reconstruct_candidate_weight_history(
    strategy_name: str,
    *,
    baseline_run: BaselineRun,
    warehouse: TradingWarehouse,
    bot_config: object | None,
) -> pd.DataFrame:
    if bot_config is None or not hasattr(bot_config, "execution"):
        return pd.DataFrame()
    candidates = warehouse.read_table("experiment_candidates")
    if candidates.empty or "strategy" not in candidates or "strategy_json" not in candidates:
        return pd.DataFrame()
    matches = candidates[candidates["strategy"].astype(str) == str(strategy_name)]
    if matches.empty:
        return pd.DataFrame()
    sort_columns = [column for column in ["iteration", "created_at_utc"] if column in matches]
    row = (
        matches.sort_values(sort_columns).iloc[-1]
        if sort_columns
        else matches.iloc[-1]
    )
    try:
        strategy = strategy_from_catalog_row(row)
        execution = execution_for_catalog_row(row, bot_config.execution)
        result, _missing = build_approach_backtest_result(
            baseline_run.prices,
            strategy,
            execution,
            scenario_sizing=scenario_sizing_from_catalog_row(row),
            future_state_model=future_state_model_from_catalog_row(row),
            strategy_drawdown_model=strategy_drawdown_model_from_catalog_row(row),
            decision_sanity=decision_sanity_from_catalog_row(row),
            name=strategy_name,
        )
    except Exception:
        return pd.DataFrame()
    if result is None:
        return pd.DataFrame()
    return _normalize_weight_history(result.weights)


def _forward_weight_history_from_valuations(
    valuations: pd.DataFrame,
    *,
    window_id: str,
) -> pd.DataFrame:
    if valuations.empty or "latest_weights_json" not in valuations:
        return pd.DataFrame()
    rows = valuations[valuations["window_id"].astype(str) == str(window_id)].copy()
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, float | pd.Timestamp]] = []
    for _, row in rows.sort_values(["valuation_date", "created_at_utc"]).iterrows():
        raw_weights = _parse_weights_json(row.get("latest_weights_json"))
        valuation_date = pd.to_datetime(row.get("valuation_date"), errors="coerce")
        if pd.isna(valuation_date) or not raw_weights:
            continue
        records.append({"date": valuation_date, **raw_weights})
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).set_index("date").sort_index()
    return _normalize_weight_history(frame)


def _parse_weights_json(raw: object) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        values = raw
    else:
        try:
            values = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(values, dict):
        return {}
    parsed: dict[str, float] = {}
    for ticker, weight in values.items():
        try:
            numeric = float(weight)
        except (TypeError, ValueError):
            continue
        if abs(numeric) > 1e-8:
            parsed[str(ticker)] = numeric
    return parsed


def _normalize_weight_history(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    frame = weights.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()].sort_index()
    if frame.empty:
        return pd.DataFrame()
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame = frame.clip(lower=0.0)
    frame = frame.loc[:, frame.abs().sum(axis=0) > 1e-8]
    frame.index.name = "date"
    return frame


def _prepare_allocation_frames(
    historical_weights: pd.DataFrame,
    forward_weights: pd.DataFrame,
    *,
    max_assets: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical_with_cash = _add_cash_residual(historical_weights)
    forward_with_cash = _add_cash_residual(forward_weights)
    combined = pd.concat([historical_with_cash, forward_with_cash], axis=0).fillna(0.0)
    if combined.empty:
        return historical_with_cash, forward_with_cash
    importance = combined.max(axis=0).sort_values(ascending=False)
    keep = [
        column
        for column in ["cash_or_unallocated"]
        if column in importance.index
    ]
    keep.extend(
        [
            column
            for column in importance.index
            if column not in keep
        ][:max(max_assets - len(keep), 0)]
    )
    return (
        _compact_to_columns(historical_with_cash, keep),
        _compact_to_columns(forward_with_cash, keep),
    )


def _add_cash_residual(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    frame = weights.copy()
    residual = (1.0 - frame.sum(axis=1)).clip(lower=0.0)
    if float(residual.max()) > 0.005:
        frame["cash_or_unallocated"] = residual
    return frame


def _compact_to_columns(weights: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    selected = [column for column in keep if column in weights]
    frame = weights[selected].copy() if selected else pd.DataFrame(index=weights.index)
    hidden = [column for column in weights.columns if column not in selected]
    if hidden:
        other = weights[hidden].sum(axis=1)
        if float(other.max()) > 0.005:
            frame["other_or_cash"] = frame.get("other_or_cash", 0.0) + other
    return frame.loc[:, frame.abs().sum(axis=0) > 1e-8]


def _apply_allocation_view_window(
    historical_weights: pd.DataFrame,
    forward_weights: pd.DataFrame,
    *,
    view: str,
    start_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pd.isna(start_date):
        return historical_weights, forward_weights
    if view == "5Y before start + forward":
        historical_weights = _filter_weight_history(
            historical_weights,
            start=start_date - pd.DateOffset(years=5),
        )
    elif view == "1Y before start + forward":
        historical_weights = _filter_weight_history(
            historical_weights,
            start=start_date - pd.DateOffset(years=1),
        )
    elif view == "Forward overlap":
        historical_weights = _filter_weight_history(historical_weights, start=start_date)
    elif view == "Forward records only":
        historical_weights = pd.DataFrame()
    return historical_weights, forward_weights


def _filter_weight_history(
    weights: pd.DataFrame,
    *,
    start: object | None = None,
    end: object | None = None,
) -> pd.DataFrame:
    if weights.empty:
        return weights
    frame = weights.copy()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end)]
    return frame


def _allocation_date_bounds(
    historical_weights: pd.DataFrame,
    forward_weights: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    indexes = [
        frame.index
        for frame in [historical_weights, forward_weights]
        if not frame.empty
    ]
    if not indexes:
        return None, None
    combined = indexes[0]
    for index in indexes[1:]:
        combined = combined.union(index)
    return pd.Timestamp(combined.min()), pd.Timestamp(combined.max())


def _clamp_date_value(
    value: object,
    *,
    min_value: date,
    max_value: date,
    fallback: date,
) -> date:
    try:
        parsed = pd.Timestamp(value).date()
    except (TypeError, ValueError):
        parsed = fallback
    if parsed < min_value or parsed > max_value:
        return fallback
    return parsed


def _make_forward_allocation_history_figure(
    historical_weights: pd.DataFrame,
    forward_weights: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    strategy_name: str,
) -> go.Figure:
    row_titles = ["Backtested target allocation", "Forward-tested allocation"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        subplot_titles=row_titles,
    )
    columns = list(dict.fromkeys([*historical_weights.columns, *forward_weights.columns]))
    palette = [
        "#0f766e",
        "#f59e0b",
        "#2563eb",
        "#dc2626",
        "#7c3aed",
        "#16a34a",
        "#0891b2",
        "#db2777",
        "#64748b",
        "#a16207",
        "#334155",
    ]
    color_lookup = {column: palette[index % len(palette)] for index, column in enumerate(columns)}
    for column in columns:
        if column in historical_weights:
            fig.add_trace(
                go.Scatter(
                    x=historical_weights.index,
                    y=historical_weights[column],
                    mode="lines",
                    name=column,
                    legendgroup=column,
                    line={"width": 0.5, "color": color_lookup[column]},
                    stackgroup="historical",
                    hovertemplate=f"{column}<br>%{{x|%Y-%m-%d}}<br>%{{y:.1%}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if column in forward_weights:
            fig.add_trace(
                go.Scatter(
                    x=forward_weights.index,
                    y=forward_weights[column],
                    mode="lines",
                    name=f"{column} forward",
                    legendgroup=column,
                    line={"width": 0.5, "color": color_lookup[column]},
                    stackgroup="forward",
                    showlegend=column not in historical_weights,
                    hovertemplate=(
                        f"{column} forward<br>%{{x|%Y-%m-%d}}<br>%{{y:.1%}}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
    if not pd.isna(start_date):
        marker_x = pd.Timestamp(start_date)
        for row, showlegend in ((1, True), (2, False)):
            fig.add_trace(
                go.Scatter(
                    x=[marker_x, marker_x],
                    y=[0, 1],
                    mode="lines",
                    name="Forward test start",
                    line={"dash": "dash", "color": "#ef4444", "width": 2},
                    showlegend=showlegend,
                    hovertemplate="Forward test start<br>%{x|%Y-%m-%d}<extra></extra>",
                ),
                row=row,
                col=1,
            )
    fig.update_layout(
        title=f"Allocation history: {strategy_name}",
        height=650,
        margin={"l": 20, "r": 20, "t": 80, "b": 30},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
        },
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title_text="Weight", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title_text="Weight", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def _latest_weight_table(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=["ticker", "target_weight"])
    latest = weights.iloc[-1].sort_values(ascending=False)
    latest = latest[latest.abs() > 1e-8]
    return pd.DataFrame(
        [{"ticker": str(ticker), "target_weight": float(weight)} for ticker, weight in latest.items()]
    )


def _format_percent_value(value: float) -> str:
    return f"{value:.1%}"


def _format_date_for_metric(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return "n/a"
    return str(value.date())


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
