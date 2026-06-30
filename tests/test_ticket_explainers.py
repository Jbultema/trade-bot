from __future__ import annotations

from trade_bot.dashboard.components import _lookup_guide_frame
from trade_bot.dashboard.ticket_explainers import (
    ticket_column_help,
    ticket_detail,
    ticket_guide_frame,
    ticket_help,
)


def test_ticket_help_resolves_ticket_fields_and_workflow_terms() -> None:
    ticket_help_text = ticket_help("ticket_id")
    forward_test_help = ticket_help("Forward Test")
    spy_detail = ticket_detail("SPY")

    assert ticket_help_text is not None
    assert "Unique local identifier" in ticket_help_text
    assert forward_test_help is not None
    assert "locks recommendations" in forward_test_help
    assert spy_detail is not None
    assert spy_detail.kind == "Ticker"


def test_ticket_guide_can_search_tickers_and_ticket_fields() -> None:
    ticker_frame = ticket_guide_frame(search="BIL")
    dollar_frame = ticket_guide_frame(search="dollar_high")

    assert not ticker_frame.empty
    assert ticker_frame.iloc[0]["term"] == "BIL"
    assert "defensive cash" in str(ticker_frame.iloc[0]["plain_english"]).lower()
    assert not dollar_frame.empty
    assert "Size Band" in set(dollar_frame["term"])


def test_generated_ticker_lookup_covers_ai_semiconductor_proxies() -> None:
    smh_detail = ticket_detail("SMH")
    soxx_detail = ticket_detail("SOXX")

    assert smh_detail is not None
    assert smh_detail.kind == "Ticker"
    assert "AI/growth" in smh_detail.plain_english
    assert soxx_detail is not None
    assert soxx_detail.kind == "Ticker"


def test_combined_lookup_prefers_exact_ticker_over_metric_text_match() -> None:
    smh_frame = _lookup_guide_frame(search="SMH")
    soxx_frame = _lookup_guide_frame(search="SOXX")

    assert smh_frame.iloc[0]["term"] == "SMH"
    assert smh_frame.iloc[0]["kind"] == "Ticker"
    assert "AI Beta" in set(smh_frame["term"])
    assert soxx_frame.iloc[0]["term"] == "SOXX"
    assert soxx_frame.iloc[0]["kind"] == "Ticker"


def test_ticket_column_help_exposes_journal_columns() -> None:
    help_by_column = ticket_column_help()

    assert "ticket_id" in help_by_column
    assert "limit_low" in help_by_column
    assert "max_shares" in help_by_column
    assert "review" in help_by_column["limit_low"].lower()
