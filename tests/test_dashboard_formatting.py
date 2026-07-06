from __future__ import annotations

import pandas as pd

from trade_bot.dashboard.formatting import _display_metrics, _escape_markdown_dollars


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


def test_escape_markdown_dollars_keeps_currency_prose_out_of_math_mode() -> None:
    text = _escape_markdown_dollars(
        "$5,062,993.91 to $8,347,478.76 against Hold QQQ median $5,090,183.54."
    )

    assert text == (
        r"\$5,062,993.91 to \$8,347,478.76 against Hold QQQ median "
        r"\$5,090,183.54."
    )
