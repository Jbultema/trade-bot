from __future__ import annotations

from trade_bot.dashboard.briefs import (
    _brief_card_html,
    _operating_card_html,
    _posture_calibration_sentence,
)


def test_operating_card_html_is_compact_and_escaped() -> None:
    html = _operating_card_html(
        {
            "tone": "warning",
            "label": "Recommended <Action>",
            "answer": "Review & Hold",
            "detail": "Move only after review.",
        }
    )

    assert html.startswith('<div class="operating-card operating-card-warning">')
    assert "\n" not in html
    assert "Recommended &lt;Action&gt;" in html
    assert "Review &amp; Hold" in html


def test_posture_calibration_sentence_explains_opportunity_pressure() -> None:
    card = _posture_calibration_sentence(
        {
            "posture_calibration_status": "opportunity_cost_watch",
            "posture_calibration_signal": "Opportunity-cost watch",
            "posture_calibration_note": "Constructive evidence still deserves review.",
            "current_risk_asset_weight": 1.0,
            "target_risk_asset_weight": 0.9,
            "one_month_risk_on_probability": 0.55,
            "constructive_scenario_probability": 0.59,
            "opportunity_pressure": 0.47,
        }
    )

    assert card["tone"] == "warning"
    assert card["answer"] == "Opportunity-cost watch"
    assert "Current risk assets 100.00%" in card["detail"]
    assert "opportunity pressure 47.00%" in card["detail"]


def test_brief_card_html_is_compact_and_escaped() -> None:
    html = _brief_card_html(
        {
            "tone": "warning",
            "label": "Risk <Rationale>",
            "answer": "Check & Review",
            "detail": "Portfolio constraints are active.",
        }
    )

    assert html.startswith('<div class="brief-card brief-card-warning">')
    assert "\n" not in html
    assert "Risk &lt;Rationale&gt;" in html
    assert "Check &amp; Review" in html
