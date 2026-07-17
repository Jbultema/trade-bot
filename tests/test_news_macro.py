from __future__ import annotations

import pandas as pd

from trade_bot.dashboard.news_macro import (
    _current_event_rollup,
    _dedupe_display_rows,
    _driver_rotation_heatmap_figure,
    _driver_rotation_scatter_figure,
)


def test_dedupe_display_rows_removes_exact_display_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {"event_id": "event_a", "title": "Same", "window": "post_5d", "extra": 1},
            {"event_id": "event_a", "title": "Same", "window": "post_5d", "extra": 2},
            {"event_id": "event_a", "title": "Same", "window": "post_21d", "extra": 3},
        ]
    )

    display = _dedupe_display_rows(frame, ["event_id", "title", "window"])

    assert len(display) == 2
    assert set(display["window"]) == {"post_5d", "post_21d"}


def test_current_event_rollup_keeps_one_row_per_event_and_counts_scenarios() -> None:
    scenarios = pd.DataFrame(
        [
            {
                "event_id": "event_a",
                "event_name": "Event A",
                "event_date": "2026-06-17",
                "category": "private_credit",
                "direction": "escalation",
                "event_phase": "leading_warning",
                "confirmation_window": "Watch credit.",
                "scenario": "Stress contained",
                "risk_posture": "Hold but watch.",
            },
            {
                "event_id": "event_a",
                "event_name": "Event A",
                "event_date": "2026-06-17",
                "category": "private_credit",
                "direction": "escalation",
                "event_phase": "leading_warning",
                "confirmation_window": "Watch credit.",
                "scenario": "Stress leaks",
                "risk_posture": "Reduce risk.",
            },
            {
                "event_id": "event_b",
                "event_name": "Event B",
                "event_date": "2026-06-16",
                "category": "ai_unit_economics",
                "direction": "escalation",
                "event_phase": "leading_warning",
                "confirmation_window": "Watch AI beta.",
                "scenario": "Ignored",
                "risk_posture": "Do not fight leadership.",
            },
        ]
    )

    rollup = _current_event_rollup(scenarios)

    assert len(rollup) == 2
    event_a = rollup[rollup["event_id"] == "event_a"].iloc[0]
    assert event_a["scenario_count"] == 2
    assert "Stress contained" in event_a["scenarios"]
    assert "Stress leaks" in event_a["scenarios"]


def test_driver_rotation_figures_render_core_traces() -> None:
    rotation = pd.DataFrame(
        [
            {
                "driver_label": "Credit conditions",
                "model_role": "allocation_driver",
                "primary_rotation_state": "normally_important_active",
                "proven_relevance": 0.8,
                "current_activation": 0.7,
                "previous_30d_activation": 0.3,
                "previous_90d_activation": 0.2,
                "change_30d": 0.4,
                "change_90d": 0.5,
                "data_support": "validated_market_or_macro_proxy",
            },
            {
                "driver_label": "AI capex pressure",
                "model_role": "explainer_only",
                "primary_rotation_state": "emerging_importance",
                "proven_relevance": 0.2,
                "current_activation": 0.8,
                "previous_30d_activation": 0.0,
                "previous_90d_activation": 0.0,
                "change_30d": 0.8,
                "change_90d": 0.8,
                "data_support": "thin_proxy",
            },
            {
                "driver_label": "Volatility",
                "model_role": "validated_context",
                "primary_rotation_state": "fading_importance",
                "proven_relevance": 0.6,
                "current_activation": 0.2,
                "previous_30d_activation": 0.7,
                "previous_90d_activation": 0.8,
                "change_30d": -0.5,
                "change_90d": -0.6,
                "data_support": "validated_market_or_macro_proxy",
            },
        ]
    )

    scatter = _driver_rotation_scatter_figure(rotation)
    heatmap = _driver_rotation_heatmap_figure(rotation)

    assert len(scatter.data) == 3
    assert len(heatmap.data) == 1
    assert "Historical Relevance" in str(scatter.layout.title.text)
    assert all(trace.mode == "markers" for trace in scatter.data)
    permanent_label_annotations = [
        annotation
        for annotation in scatter.layout.annotations
        if annotation.showarrow is False and annotation.text in {"Credit conditions", "AI capex pressure"}
    ]
    movement_arrows = [
        annotation
        for annotation in scatter.layout.annotations
        if annotation.showarrow is True
    ]
    assert not permanent_label_annotations
    assert {annotation.arrowcolor for annotation in movement_arrows} == {"#16a34a", "#dc2626"}
    assert all(float(annotation.arrowwidth) >= 3.0 for annotation in movement_arrows)
