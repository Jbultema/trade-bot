from __future__ import annotations

import pandas as pd

from trade_bot.research.capital_cycle_targets import build_objective_economic_targets


def _episode_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode": "housing",
                "episode_family": "housing",
                "episode_type": "break",
                "start_date": "2018-01-01",
                "end_date": "2021-12-31",
                "theme_ticker": "XHB",
                "preregistered_at": "2026-08-19",
            }
        ]
    )


def _target_spec() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode": "housing",
                "target_component": "housing_starts_drawdown",
                "family": "demand",
                "direction": "lower_is_rollover",
                "lookback_periods": 3,
                "threshold": -0.20,
                "persistence_periods": 2,
                "minimum_independent_components": 1,
                "preregistered_at": "2026-08-19",
            }
        ]
    )


def _observations() -> pd.DataFrame:
    values = [100.0, 110.0, 120.0, 115.0, 85.0, 80.0]
    rows = []
    for index, value in enumerate(values):
        period_end = pd.Timestamp("2018-03-31") + pd.offsets.QuarterEnd(index)
        rows.append(
            {
                "episode": "housing",
                "family": "demand",
                "metric": "housing_starts_saar_thousands",
                "entity": "US",
                "period_end": period_end,
                "available_date": period_end + pd.Timedelta(days=20),
                "value": value,
                "source_id": f"release_{index}",
                "source_url": "https://www.census.gov/archive.pdf",
                "source_tier": "A",
                "vintage_status": "archived_release",
                "eligible_for_model": True,
            }
        )
    return pd.DataFrame(rows)


def test_objective_targets_use_distinct_release_persistence(tmp_path) -> None:
    result = build_objective_economic_targets(
        episode_manifest=_episode_manifest(),
        observations=_observations(),
        target_spec=_target_spec(),
        output_dir=tmp_path,
    )
    event = result.event_manifest.iloc[0]
    assert event["objective_label_resolved"]
    assert pd.notna(event["objective_rollover_date"])
    flagged = result.component_panel[result.component_panel["persistent_component_flag"]]
    assert not flagged.empty
    assert flagged.iloc[0]["evidence_updated_at_origin"]


def test_missing_component_history_keeps_target_unresolved(tmp_path) -> None:
    result = build_objective_economic_targets(
        episode_manifest=_episode_manifest(),
        observations=_observations().iloc[:2],
        target_spec=_target_spec(),
        output_dir=tmp_path,
    )
    assert not result.event_manifest.iloc[0]["objective_label_resolved"]
    assert not result.readiness.iloc[0]["economic_target_gate_passed"]
