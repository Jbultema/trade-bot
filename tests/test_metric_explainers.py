from __future__ import annotations

from trade_bot.dashboard.metric_explainers import metric_detail, metric_guide_frame, metric_help


def test_metric_help_resolves_labels_and_column_aliases() -> None:
    cagr_help = metric_help("CAGR")
    walk_forward_help = metric_help("walk_forward_positive_rate")

    assert cagr_help is not None
    assert "Compounded annual growth rate" in cagr_help
    assert walk_forward_help is not None
    assert "walk-forward" in walk_forward_help.lower()


def test_metric_guide_can_filter_and_search() -> None:
    frame = metric_guide_frame(category="Risk Engine", search="expected shortfall")

    assert not frame.empty
    assert set(frame["category"]) == {"Risk Engine"}
    assert "Expected Shortfall 95" in set(frame["metric"])
    assert metric_detail("ES 95") is not None


def test_metric_help_resolves_outcome_simulation_terms() -> None:
    assert "Outcome Frontier" in (metric_help("Starting Account") or "")
    assert "Sequence-aware wealth distribution" in (metric_help("Bootstrap P10 Wealth") or "")
    assert metric_detail("Median Sim Ulcer") is not None
