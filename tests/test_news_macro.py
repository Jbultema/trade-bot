from __future__ import annotations

import pandas as pd

from trade_bot.dashboard.news_macro import _current_event_rollup, _dedupe_display_rows


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
