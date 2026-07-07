from __future__ import annotations

from trade_bot.dashboard.components import _lookup_guide_frame
from trade_bot.dashboard.navigation import dashboard_section_names
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


def test_combined_lookup_covers_launch_lab_terms() -> None:
    expected_terms = {
        "beat rate": "Beat Rate",
        "bad-start rate": "Bad-Start Rate",
        "positive-start rate": "Positive-Start Rate",
        "Launch Gate": "Launch Readiness",
        "starter sleeve": "Ramp Plan",
        "first month drawdown": "First-Month Drawdown",
        "final sleeve fraction": "Final Sleeve Fraction",
        "entry backtest": "Entry Backtest",
        "test capital": "Test Capital",
        "current blockers": "Launch Readiness",
        "why keep watching": "Launch Readiness",
        "what to do instead": "Launch Readiness",
        "entry friction": "Launch Readiness",
        "recommended protocol": "Launch Protocol",
        "historical pattern": "Entry Backtest",
        "best start": "Entry Backtest",
        "worst start": "Entry Backtest",
        "ramp weeks": "Ramp Plan",
        "reserved cash": "Ramp Plan",
        "new capital": "Launching vs Operating",
        "running book": "Launching vs Operating",
        "scale-up capital": "Launching vs Operating",
        "benchmark return": "Launch Benchmark",
        "total return": "Entry Backtest",
    }

    for query, expected_term in expected_terms.items():
        frame = _lookup_guide_frame(search=query)

        assert not frame.empty, query
        assert expected_term in set(frame["term"]), query
        assert "Metric" in set(frame.loc[frame["term"].eq(expected_term), "kind"]), query


def test_combined_lookup_prioritizes_exact_launch_metric() -> None:
    frame = _lookup_guide_frame(search="beat rate")

    assert frame.iloc[0]["term"] == "Beat Rate"
    assert frame.iloc[0]["kind"] == "Metric"


def test_combined_lookup_matches_human_spacing_to_snake_case_terms() -> None:
    expected_terms = {
        "score impact": "Current Launch Diagnostics",
        "risk off 1m probability": "Current Launch Diagnostics",
        "first month drawdown": "First-Month Drawdown",
        "capital deployed": "Ramp Plan",
        "ticket id": "Ticket ID",
    }

    for query, expected_term in expected_terms.items():
        frame = _lookup_guide_frame(search=query)

        assert not frame.empty, query
        assert expected_term in set(frame["term"]), query


def test_combined_lookup_covers_dashboard_workbench_terms() -> None:
    for section_name in dashboard_section_names():
        frame = _lookup_guide_frame(search=section_name)

        assert not frame.empty, section_name


def test_ticket_column_help_exposes_journal_columns() -> None:
    help_by_column = ticket_column_help()

    assert "ticket_id" in help_by_column
    assert "limit_low" in help_by_column
    assert "max_shares" in help_by_column
    assert "review" in help_by_column["limit_low"].lower()
