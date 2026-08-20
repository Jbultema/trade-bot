from __future__ import annotations

import pandas as pd

from trade_bot.research.capital_cycle_targets import (
    build_housing_threshold_sensitivity,
    build_objective_economic_targets,
)


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


def test_episode_without_target_spec_fails_closed_instead_of_crashing(tmp_path) -> None:
    episodes = pd.concat(
        [
            _episode_manifest(),
            pd.DataFrame(
                [
                    {
                        "episode": "unconfigured",
                        "episode_family": "other",
                        "episode_type": "break",
                        "start_date": "2018-01-01",
                        "end_date": "2021-12-31",
                        "theme_ticker": "QQQ",
                        "preregistered_at": "2026-08-20",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = build_objective_economic_targets(
        episode_manifest=episodes,
        observations=_observations(),
        target_spec=_target_spec(),
        output_dir=tmp_path,
    )

    event = result.event_manifest.set_index("episode").loc["unconfigured"]
    assert not event["target_spec_registered"]
    assert not event["objective_label_resolved"]
    assert "unconfigured" in result.readiness.iloc[0]["unresolved_episodes"]


def test_housing_sensitivity_is_bounded_exploratory_and_has_no_authority() -> None:
    episodes = pd.DataFrame(
        [
            {
                "episode": "housing_mortgage_2001_2011",
                "first_deterioration_date": "2006-01-01",
                "market_recognition_date": "2007-02-27",
            },
            {
                "episode": "housing_recovery_2012_2019",
                "first_deterioration_date": pd.NaT,
                "market_recognition_date": pd.NaT,
            },
        ]
    )
    rows = []
    for episode, origin, starts, supply, credit in (
        ("housing_mortgage_2001_2011", "2006-09-01", -0.16, 0.11, 0.00),
        ("housing_mortgage_2001_2011", "2006-10-01", -0.17, 0.12, 0.00),
        ("housing_mortgage_2001_2011", "2006-11-01", -0.18, 0.13, 0.00),
        ("housing_recovery_2012_2019", "2015-01-01", 0.05, 0.02, 0.00),
        ("housing_recovery_2012_2019", "2015-02-01", 0.04, 0.03, 0.00),
        ("housing_recovery_2012_2019", "2015-03-01", 0.03, 0.04, 0.00),
    ):
        for component, value, direction, persistence in (
            ("housing_starts_drawdown", starts, "lower_is_rollover", 3),
            ("months_supply_increase", supply, "higher_is_rollover", 3),
            ("mortgage_credit_tightening", credit, "higher_is_rollover", 2),
        ):
            rows.append(
                {
                    "episode": episode,
                    "origin_date": origin,
                    "target_component": component,
                    "direction": direction,
                    "persistence_periods": persistence,
                    "evidence_updated_at_origin": True,
                    "transformed_value": value,
                }
            )

    sensitivity = build_housing_threshold_sensitivity(pd.DataFrame(rows), episodes)

    assert len(sensitivity) == 18
    assert sensitivity["negative_control_passed"].all()
    assert sensitivity["allocation_authority"].eq(0.0).all()
    assert sensitivity["analysis_role"].eq(
        "post_hoc_exploratory_sensitivity_only"
    ).all()
    assert sensitivity["preregistered_configuration"].sum() == 1
    early = sensitivity[
        sensitivity["housing_starts_drawdown"].eq(-0.15)
        & sensitivity["months_supply_increase"].eq(0.10)
    ].iloc[0]
    assert early["break_event_date"] == pd.Timestamp("2006-11-01")
