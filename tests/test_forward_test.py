from __future__ import annotations

import pandas as pd

from trade_bot.dashboard.components import _metric_column_config
from trade_bot.dashboard.forward_test import (
    TICKET_COLUMN_HELP,
    _ticket_option_label,
    _ticket_option_map,
)


def test_ticket_option_label_summarizes_action_without_uuid_decoding() -> None:
    row = pd.Series(
        {
            "ticket_id": "12345678-abcd-ef00-1111-222222222222",
            "ticker": "qqq",
            "side": "buy",
            "status": "open",
            "strategy_name": "scenario_adjusted_trade_decision",
            "min_shares": 1.25,
            "max_shares": 2.5,
        }
    )

    label = _ticket_option_label(row)

    assert label == ("QQQ BUY 1.25-2.50 sh | open | " "scenario_adjusted_trade_decision | 12345678")


def test_ticket_option_map_preserves_ticket_id_lookup() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticket_id": "ticket-a",
                "ticker": "BIL",
                "side": "BUY",
                "status": "open",
                "strategy_name": "risk_off",
                "min_shares": 10,
                "max_shares": 12,
            }
        ]
    )

    options = _ticket_option_map(tickets)

    assert list(options.values()) == ["ticket-a"]
    assert list(options)[0].startswith("BIL BUY 10.00-12.00 sh")


def test_ticket_table_tooltips_override_ambiguous_columns() -> None:
    frame = pd.DataFrame(columns=["status", "ticket_id", "ticker"])

    config = _metric_column_config(frame, column_help=TICKET_COLUMN_HELP)

    assert set(config) == {"status", "ticket_id", "ticker"}
