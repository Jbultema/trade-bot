from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.capital_cycle_vintages import save_prospective_ai_vintage


def test_prospective_vintage_is_immutable_and_records_missingness(tmp_path) -> None:
    observations = pd.DataFrame(
        [
            {
                "family": "demand",
                "metric": "ai_revenue",
                "entity": "sample",
                "period_end": "2026-06-30",
                "available_date": "2026-07-30",
                "value": 1.0,
                "source_id": "filing",
                "source_url": "https://www.sec.gov/filing",
                "vintage_status": "filing_as_reported",
            },
            {
                "family": "capacity",
                "metric": "future_capex",
                "entity": "sample",
                "period_end": "2026-09-30",
                "available_date": "2026-09-30",
                "value": 2.0,
                "source_id": "future",
                "source_url": "https://example.com",
                "vintage_status": "filing_as_reported",
            },
        ]
    )
    result = save_prospective_ai_vintage(
        observations=observations,
        as_of="2026-08-19",
        output_dir=tmp_path,
    )
    assert len(result.observations) == 1
    assert int(result.missingness["observed"].sum()) == 1
    assert int(result.missingness["available"].sum()) == 0
    demand = result.missingness.set_index("family").loc["demand"]
    assert "insufficient_entity_breadth" in demand["missing_reason"]
    assert result.manifest_path.exists()
    with pytest.raises(FileExistsError):
        save_prospective_ai_vintage(
            observations=observations,
            as_of="2026-08-20",
            output_dir=tmp_path,
        )
