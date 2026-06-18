from __future__ import annotations

import html

import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_trade_frame
from trade_bot.trading.book_alignment import BookAlignmentRun

BOOK_ALIGNMENT_COLUMN_HELP = {
    "alignment_status": "Book-vs-target state for the selected paper/live account: aligned, unstarted, small_drift, rebalance_needed, or critical_rebalance.",
    "recommended_action": "Execution-layer action after comparing logged holdings to the latest target posture.",
    "book_scope": "What executions are treated as the current book of record. Account scope means all logged executions for the selected mode and account.",
    "account_value": "Account value used to translate target weights and drift into dollar sizing.",
    "current_position": "Current logged book weights inferred from executions and latest prices.",
    "target_position": "Latest scenario- and risk-adjusted target posture from the trade-decision layer.",
    "current_cash_weight": "Unallocated cash implied by account value minus current marked holdings.",
    "target_cash_weight": "Target unallocated cash implied by one minus target tradable weights.",
    "max_abs_delta": "Largest absolute target-minus-current weight gap in the selected book.",
    "material_trade_count": "Number of ticker gaps above the minimum trade-weight threshold.",
    "largest_ticker": "Ticker with the largest absolute book-vs-target drift.",
    "largest_delta_weight": "Signed target-minus-current weight for the largest drift ticker.",
    "largest_delta_notional": "Signed dollar change implied by the largest drift ticker before price and size bands.",
    "current_notional": "Current marked dollar value from logged net quantity times latest reference price.",
    "target_notional": "Dollar target from account value times target weight.",
    "delta_notional": "Dollar change needed to move current marked notional to target notional.",
    "net_quantity": "Net shares accumulated from logged executions for the selected account.",
    "net_cash_deployed": "Net historical dollars deployed by logged buys minus sells.",
}


def _render_book_alignment(
    alignment: BookAlignmentRun,
    *,
    heading: str = "Book Alignment",
    show_position_plan: bool = True,
) -> None:
    if alignment.summary.empty:
        return

    row = alignment.summary.iloc[0]
    status = str(row.get("alignment_status", "unknown"))
    action = str(row.get("recommended_action", ""))
    level = _status_level(status)
    st.subheader(heading)
    st.markdown(
        f'''
        <div class="action-banner action-{html.escape(level)}">
            <p class="headline-label">Book-Aware Recommendation</p>
            <div class="headline-title">{html.escape(_status_title(status, action))}</div>
            <p class="headline-copy">{html.escape(str(row.get("explanation", "")))}</p>
            <p class="headline-next">Scope: {html.escape(str(row.get("mode", "")))} / {html.escape(str(row.get("account", "")))} / {html.escape(str(row.get("strategy_name", "")))}.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cols[0].metric("Book Status", _status_label(status))
    cols[1].metric("Action", action.replace("_", " ").title())
    cols[2].metric("Max Drift", f"{float(row.get('max_abs_delta', 0.0)):.1%}")
    cols[3].metric("Trades", f"{int(row.get('material_trade_count', 0))}")
    cols[4].metric("Account", f"${float(row.get('account_value', 0.0)):,.0f}")

    summary_columns = [
        "mode",
        "account",
        "strategy_name",
        "book_scope",
        "alignment_status",
        "recommended_action",
        "current_position",
        "target_position",
        "current_cash_weight",
        "target_cash_weight",
        "largest_ticker",
        "largest_delta_weight",
        "largest_delta_notional",
    ]
    with st.expander("Book alignment details", expanded=False):
        available = [column for column in summary_columns if column in alignment.summary]
        _render_metric_dataframe(
            _display_trade_frame(alignment.summary[available]),
            use_container_width=True,
            hide_index=True,
            column_help=BOOK_ALIGNMENT_COLUMN_HELP,
        )

    if not show_position_plan:
        return

    st.caption("Current logged book versus latest target weights")
    if alignment.position_plan.empty:
        st.write("No book or target holdings are available for this context.")
        return
    position_columns = [
        "ticker",
        "action",
        "current_weight",
        "scenario_adjusted_weight",
        "delta_weight",
        "reference_price",
        "net_quantity",
        "current_notional",
        "target_notional",
        "delta_notional",
    ]
    available = [column for column in position_columns if column in alignment.position_plan]
    _render_metric_dataframe(
        _display_trade_frame(alignment.position_plan[available]),
        use_container_width=True,
        column_help=BOOK_ALIGNMENT_COLUMN_HELP,
    )


def _status_level(status: str) -> str:
    if status == "aligned":
        return "do_nothing"
    if status in {"critical_rebalance", "unstarted"}:
        return "critical_actions"
    return "small_actions"


def _status_title(status: str, action: str) -> str:
    if status == "aligned":
        return "Book aligned with latest target"
    if status == "unstarted":
        return "No logged book yet"
    return action.replace("_", " ").title()


def _status_label(status: str) -> str:
    return status.replace("_", " ").title()
