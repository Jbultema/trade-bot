from __future__ import annotations

import pandas as pd

from trade_bot.dashboard.formatting import _display_metrics


def test_display_metrics_formats_operability_labels() -> None:
    frame = pd.DataFrame(
        {
            "operability_label": ["weekly_large_moves", "weekly_cadence"],
            "monitoring_readiness_label": ["paper_candidate", "paper_ready"],
        }
    )

    display = _display_metrics(frame)

    assert display["operability_label"].tolist() == [
        "Weekly cadence, large moves",
        "Weekly cadence",
    ]
    assert display["monitoring_readiness_label"].tolist() == [
        "Paper candidate",
        "Paper ready",
    ]
