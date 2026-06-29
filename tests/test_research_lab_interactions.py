from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trade_bot.dashboard.research_lab import (
    _outcome_label_index_for_strategy,
    _plotly_selected_strategy,
)


def test_plotly_selected_strategy_reads_dict_event_payload() -> None:
    event = {"selection": {"points": [{"customdata": ["strategy_b", 0.95]}]}}

    assert _plotly_selected_strategy(event) == "strategy_b"


def test_plotly_selected_strategy_reads_attribute_event_payload() -> None:
    event = SimpleNamespace(
        selection=SimpleNamespace(
            points=[SimpleNamespace(customdata=("strategy_a", 0.88))]
        )
    )

    assert _plotly_selected_strategy(event) == "strategy_a"


def test_plotly_selected_strategy_reads_array_like_customdata() -> None:
    event = {"selection": {"points": [{"customdata": pd.Series(["strategy_c", 0.91])}]}}

    assert _plotly_selected_strategy(event) == "strategy_c"


def test_plotly_selected_strategy_returns_none_for_empty_selection() -> None:
    assert _plotly_selected_strategy({"selection": {"points": []}}) is None
    assert _plotly_selected_strategy({"selection": {}}) is None


def test_outcome_label_index_for_strategy_falls_back_to_first_option() -> None:
    options = pd.DataFrame(
        {
            "strategy": ["strategy_a", "strategy_b"],
            "label": ["Strategy A", "Strategy B"],
        },
        index=[10, 20],
    )

    assert _outcome_label_index_for_strategy(options, "strategy_b") == 1
    assert _outcome_label_index_for_strategy(options, "missing") == 0
