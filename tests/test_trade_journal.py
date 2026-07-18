from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.research.trade_decision import TradeDecisionRun
from trade_bot.trading.journal import (
    TicketSizingConfig,
    TradeJournal,
    build_recommendation_tickets,
)


def test_build_recommendation_tickets_from_trade_decision() -> None:
    trade_decision = _trade_decision()
    prices = pd.DataFrame(
        {
            "QQQ": [500.0],
            "BIL": [100.0],
        },
        index=pd.to_datetime(["2026-06-17"]),
    )

    tickets = build_recommendation_tickets(
        trade_decision,
        prices,
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        sizing=TicketSizingConfig(
            account_value=1000.0,
            price_band_pct=0.01,
            size_band_pct=0.20,
            whole_shares=False,
        ),
    )

    assert set(tickets["ticker"]) == {"QQQ", "BIL"}
    bil = tickets[tickets["ticker"] == "BIL"].iloc[0]
    assert bil["side"] == "BUY"
    assert bil["target_notional"] == 260.0
    assert bil["limit_low"] == 99.0
    assert bil["limit_high"] == 101.0
    assert bil["min_shares"] == 2.08
    assert bil["max_shares"] == 3.12

    qqq = tickets[tickets["ticker"] == "QQQ"].iloc[0]
    assert qqq["side"] == "SELL"
    assert qqq["target_notional"] == -130.0


def test_trade_journal_persists_snapshot_tickets_and_execution(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "journal.sqlite")
    trade_decision = _trade_decision()
    prices = pd.DataFrame(
        {
            "QQQ": [500.0],
            "BIL": [100.0],
        },
        index=pd.to_datetime(["2026-06-17"]),
    )
    sizing = TicketSizingConfig(account_value=1000.0, whole_shares=False)
    tickets = build_recommendation_tickets(
        trade_decision,
        prices,
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        sizing=sizing,
    )

    decision_id = journal.save_decision_snapshot(
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        trade_decision=trade_decision,
        sizing=sizing,
        tickets=tickets,
    )
    stored_tickets = journal.load_recommendation_tickets(status="open")
    ticket_id = str(stored_tickets.iloc[0]["ticket_id"])
    execution_id = journal.log_execution(
        mode="paper",
        account="shadow",
        ticker=str(stored_tickets.iloc[0]["ticker"]),
        side=str(stored_tickets.iloc[0]["side"]),
        quantity=1.0,
        price=float(stored_tickets.iloc[0]["reference_price"]),
        executed_at_utc="2026-06-17T16:00:00+00:00",
        recommendation_id=ticket_id,
    )

    snapshots = journal.load_decision_snapshots()
    executions = journal.load_executions()
    open_tickets = journal.load_recommendation_tickets(status="open")
    position_summary = journal.execution_position_summary()

    assert snapshots.iloc[0]["decision_id"] == decision_id
    assert executions.iloc[0]["execution_id"] == execution_id
    assert ticket_id not in set(open_tickets["ticket_id"])
    assert position_summary["net_quantity"].abs().sum() == 1.0


def test_trade_journal_manages_named_books(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "journal.sqlite")

    default_book = journal.get_promoted_book()
    assert default_book.book_name == "Default Paper Book"
    assert default_book.is_promoted

    live_book_id = journal.upsert_book(
        book_name="Live IRA",
        mode="live",
        account="vanguard_ira",
        strategy_name="scenario_adjusted_trade_decision",
        account_value=50_000.0,
        promote=True,
    )
    live_book = journal.get_promoted_book()

    assert live_book.book_id == live_book_id
    assert live_book.mode == "live"
    assert live_book.account == "vanguard_ira"
    assert live_book.account_value == 50_000.0

    journal.delete_book(default_book.book_id)
    books = journal.list_books()
    assert set(books["book_name"]) == {"Live IRA"}

    try:
        journal.delete_book(live_book_id)
    except ValueError as exc:
        assert "Promoted book cannot be deleted" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("promoted book deletion should fail")


def test_trade_journal_filters_tickets_by_book_scope(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "journal.sqlite")
    trade_decision = _trade_decision()
    prices = pd.DataFrame(
        {"QQQ": [500.0], "BIL": [100.0]},
        index=pd.to_datetime(["2026-06-17"]),
    )
    sizing = TicketSizingConfig(account_value=1000.0, whole_shares=False)

    for account in ["paper_a", "paper_b"]:
        tickets = build_recommendation_tickets(
            trade_decision,
            prices,
            mode="paper",
            account=account,
            strategy_name="scenario_adjusted_trade_decision",
            sizing=sizing,
        )
        journal.save_decision_snapshot(
            mode="paper",
            account=account,
            strategy_name="scenario_adjusted_trade_decision",
            trade_decision=trade_decision,
            sizing=sizing,
            tickets=tickets,
        )

    scoped = journal.load_recommendation_tickets(
        status="open",
        mode="paper",
        account="paper_a",
        strategy_name="scenario_adjusted_trade_decision",
    )

    assert not scoped.empty
    assert set(scoped["account"]) == {"paper_a"}


def _trade_decision() -> TradeDecisionRun:
    return TradeDecisionRun(
        summary=pd.DataFrame(
            [
                {
                    "recommended_action": "REVIEW_REDUCE_RISK",
                    "risk_status": "yellow",
                    "risk_budget_multiplier": 0.74,
                    "base_position": "QQQ 50%, BIL 0%",
                    "scenario_adjusted_position": "QQQ 37%, BIL 26%",
                    "human_explanation": "Because risk is elevated, reduce QQQ and add BIL.",
                }
            ]
        ),
        position_plan=pd.DataFrame(
            [
                {
                    "ticker": "QQQ",
                    "current_weight": 0.50,
                    "scenario_adjusted_weight": 0.37,
                    "delta_weight": -0.13,
                    "action": "REDUCE",
                },
                {
                    "ticker": "BIL",
                    "current_weight": 0.00,
                    "scenario_adjusted_weight": 0.26,
                    "delta_weight": 0.26,
                    "action": "ADD",
                },
            ]
        ),
        evidence=pd.DataFrame(
            [{"evidence_type": "risk_status", "signal": "YELLOW", "impact": "reduce risk"}]
        ),
        scenario_links=pd.DataFrame(
            [{"scenario": "Choppy rotation", "probability": 0.2, "risk_bucket": "transition"}]
        ),
    )
