from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trade_bot.research.trade_decision import TradeDecisionRun
from trade_bot.trading.book_alignment import build_book_alignment, latest_book_account_value
from trade_bot.trading.journal import (
    TicketSizingConfig,
    TradeJournal,
    build_recommendation_tickets,
)


def test_book_alignment_creates_initial_book_from_zero_executions(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "journal.sqlite")
    alignment = build_book_alignment(
        journal=journal,
        trade_decision=_trade_decision(),
        prices=_prices(),
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        account_value=1000.0,
    )

    summary = alignment.summary.iloc[0]
    assert summary["alignment_status"] == "unstarted"
    assert summary["recommended_action"] == "START_PAPER_BOOK"
    assert summary["material_trade_count"] == 2
    assert set(alignment.position_plan["action"]) == {"ADD"}

    tickets = build_recommendation_tickets(
        _trade_decision(),
        _prices(),
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        sizing=TicketSizingConfig(account_value=1000.0, whole_shares=False),
        position_plan=alignment.position_plan,
    )
    assert set(tickets["ticker"]) == {"QQQ", "BIL"}


def test_book_alignment_recognizes_executed_target_as_aligned(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "journal.sqlite")
    journal.log_execution(
        mode="paper",
        account="shadow",
        ticker="QQQ",
        side="BUY",
        quantity=0.74,
        price=500.0,
        executed_at_utc="2026-06-17T16:00:00+00:00",
    )
    journal.log_execution(
        mode="paper",
        account="shadow",
        ticker="BIL",
        side="BUY",
        quantity=2.6,
        price=100.0,
        executed_at_utc="2026-06-17T16:01:00+00:00",
    )

    alignment = build_book_alignment(
        journal=journal,
        trade_decision=_trade_decision(),
        prices=_prices(),
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        account_value=1000.0,
    )

    summary = alignment.summary.iloc[0]
    assert summary["alignment_status"] == "aligned"
    assert summary["recommended_action"] == "DO_NOTHING"
    assert summary["material_trade_count"] == 0
    assert alignment.position_plan["delta_weight"].abs().max() < 1e-12

    tickets = build_recommendation_tickets(
        _trade_decision(),
        _prices(),
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        sizing=TicketSizingConfig(account_value=1000.0, whole_shares=False),
        position_plan=alignment.position_plan,
    )
    assert tickets.empty


def test_book_alignment_floors_account_value_at_marked_holdings(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "journal.sqlite")
    journal.log_execution(
        mode="paper",
        account="shadow",
        ticker="QQQ",
        side="BUY",
        quantity=10.0,
        price=500.0,
        executed_at_utc="2026-06-17T16:00:00+00:00",
    )

    alignment = build_book_alignment(
        journal=journal,
        trade_decision=_trade_decision(),
        prices=_prices(),
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        account_value=1000.0,
    )

    summary = alignment.summary.iloc[0]
    qqq = alignment.position_plan.set_index("ticker").loc["QQQ"]
    assert summary["account_value"] == 5000.0
    assert summary["account_value_input"] == 1000.0
    assert summary["account_value_source"] == "marked_holdings_floor"
    assert "Logged holdings exceed" in summary["account_value_warning"]
    assert qqq["current_weight"] == 1.0
    assert float(alignment.position_plan["current_weight"].abs().max()) <= 1.0


def test_latest_saved_account_value_preserves_cash_after_partial_sale(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "journal.sqlite")
    base_decision = _trade_decision()
    decision = TradeDecisionRun(
        summary=base_decision.summary,
        position_plan=pd.DataFrame(
            [
                {
                    "ticker": "QQQ",
                    "current_weight": 0.50,
                    "scenario_adjusted_weight": 0.50,
                    "delta_weight": 0.00,
                    "action": "HOLD",
                },
                {
                    "ticker": "BIL",
                    "current_weight": 0.50,
                    "scenario_adjusted_weight": 0.50,
                    "delta_weight": 0.00,
                    "action": "HOLD",
                },
            ]
        ),
        evidence=pd.DataFrame(),
        scenario_links=pd.DataFrame(),
    )
    journal.save_decision_snapshot(
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        trade_decision=decision,
        sizing=TicketSizingConfig(account_value=1000.0, whole_shares=False),
        tickets=pd.DataFrame(),
    )
    journal.log_execution(
        mode="paper",
        account="shadow",
        ticker="QQQ",
        side="BUY",
        quantity=1.0,
        price=500.0,
        executed_at_utc="2026-06-17T16:00:00+00:00",
    )
    journal.log_execution(
        mode="paper",
        account="shadow",
        ticker="BIL",
        side="BUY",
        quantity=5.0,
        price=100.0,
        executed_at_utc="2026-06-17T16:01:00+00:00",
    )
    journal.log_execution(
        mode="paper",
        account="shadow",
        ticker="BIL",
        side="SELL",
        quantity=2.0,
        price=100.0,
        executed_at_utc="2026-06-17T16:02:00+00:00",
    )

    account_value = latest_book_account_value(
        journal,
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        default=100.0,
    )
    alignment = build_book_alignment(
        journal=journal,
        trade_decision=decision,
        prices=_prices(),
        mode="paper",
        account="shadow",
        strategy_name="scenario_adjusted_trade_decision",
        account_value=account_value,
    )

    summary = alignment.summary.iloc[0]
    positions = alignment.position_plan.set_index("ticker")
    assert account_value == 1000.0
    assert summary["account_value"] == 1000.0
    assert summary["account_value_source"] == "input"
    assert summary["current_cash_weight"] == pytest.approx(0.2)
    assert positions.loc["QQQ", "current_weight"] == 0.5
    assert positions.loc["QQQ", "delta_weight"] == 0.0
    assert positions.loc["QQQ", "action"] == "HOLD"
    assert positions.loc["BIL", "current_weight"] == 0.3
    assert positions.loc["BIL", "delta_weight"] == 0.2
    assert positions.loc["BIL", "action"] == "ADD"


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "QQQ": [500.0],
            "BIL": [100.0],
        },
        index=pd.to_datetime(["2026-06-17"]),
    )


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
        evidence=pd.DataFrame(),
        scenario_links=pd.DataFrame(),
    )
