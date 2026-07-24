from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trade_bot.dashboard.research_lab import (
    _outcome_label_index_for_strategy,
    _outcome_metric_peer_context,
    _peer_percentile,
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
    assert _plotly_selected_strategy({"selection": {"points": [{"customdata": [""]}]}}) is None


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


def test_peer_percentile_respects_metric_direction() -> None:
    values = pd.Series([0.05, 0.10, 0.15, 0.20])

    assert _peer_percentile(0.15, values, lower_is_better=False) == 0.75
    assert _peer_percentile(0.10, values, lower_is_better=True) == 0.75


def test_outcome_peer_context_fails_closed_for_missing_optional_metric(
    monkeypatch,
) -> None:
    working = pd.DataFrame(
        {
            "strategy": ["runtime_snapshot", "peer"],
            "is_selected": [True, False],
            "cagr": [0.20, 0.12],
            "max_drawdown": [-0.25, -0.30],
        }
    )
    monkeypatch.setattr(
        "trade_bot.dashboard.research_lab._outcome_peer_distribution_frame",
        lambda *args, **kwargs: working,
    )

    context = _outcome_metric_peer_context(
        pd.Series({"strategy": "runtime_snapshot"}),
        selected_result=None,
        selected_ulcer_index=None,
        selected_underwater_rate=None,
        peer_frame=pd.DataFrame(),
        bot_config=SimpleNamespace(),
        baseline_run=SimpleNamespace(),
        experiment_scorecards=pd.DataFrame(),
        benchmark_values={},
    )

    left_tail = context.loc[context["metric"].eq("Left-Tail Regime Return")].iloc[0]
    assert left_tail["selected"] == "not available"
    assert left_tail["peer_count"] == 0
