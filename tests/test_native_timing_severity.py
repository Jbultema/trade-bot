from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trade_bot.research import native_timing_severity as severity


def _inputs() -> dict[str, pd.Series]:
    index = pd.date_range("2025-01-01", periods=3)
    return {
        "base_defense": pd.Series(0.60, index=index),
        "mild_probability": pd.Series(0.20, index=index),
        "break_count": pd.Series([1.0, 2.0, 3.0], index=index),
        "spy_drawdown": pd.Series([-0.02, -0.06, -0.12], index=index),
        "realized_volatility": pd.Series([0.15, 0.25, 0.30], index=index),
        "high_volatility_threshold": pd.Series(0.20, index=index),
        "non_deterioration_age": pd.Series(0.0, index=index),
    }


def test_joint_severe_restoration_requires_all_three_states() -> None:
    inputs = _inputs()
    mild = severity.severity_policy_target(policy="mild_frozen", **inputs)
    restored = severity.severity_policy_target(policy="joint_severe_restore_10", **inputs)

    assert restored.iloc[0] == pytest.approx(mild.iloc[0])
    assert restored.iloc[1] - mild.iloc[1] == pytest.approx(0.10)
    assert restored.iloc[2] - mild.iloc[2] == pytest.approx(0.10)


def test_extreme_veto_does_not_activate_before_deep_confirmed_state() -> None:
    inputs = _inputs()
    mild = severity.severity_policy_target(policy="mild_frozen", **inputs)
    veto = severity.severity_policy_target(policy="extreme_native_veto", **inputs)

    assert veto.iloc[0] == pytest.approx(mild.iloc[0])
    assert veto.iloc[1] == pytest.approx(mild.iloc[1])
    assert veto.iloc[2] == pytest.approx(inputs["base_defense"].iloc[2])


def test_activation_episodes_do_not_treat_separated_states_as_one_event() -> None:
    dates = pd.date_range("2025-01-01", periods=5, freq="W-WED")
    predictions = pd.DataFrame(
        {
            "family": ["i111"] * 5,
            "origin_date": dates,
            "break_count": [2.0, 2.0, 0.0, 2.0, 0.0],
            "spy_drawdown_252": [-0.06, -0.07, -0.02, -0.08, -0.01],
            "realized_vol_21": [0.30, 0.31, 0.10, 0.32, 0.10],
            "training_realized_vol_q75": [0.20] * 5,
            "forward_worst_drawdown": [-0.10, -0.11, -0.02, -0.12, -0.01],
            "forward_break": [1.0, 1.0, 0.0, 1.0, 0.0],
        }
    )

    episodes = severity.build_severity_activation_episodes(predictions)

    assert len(episodes) == 2
    assert episodes["active_weeks"].tolist() == [2, 1]
    assert episodes["share_of_active_weeks"].max() == pytest.approx(2.0 / 3.0)


def test_freeze_shadow_augments_schema_without_rewriting_frozen_values(
    tmp_path: Path,
) -> None:
    existing = pd.DataFrame(
        [
            {
                "candidate": "candidate",
                "status": "prospective_observation_only",
                "retrospective_focus_cagr_delta_vs_native": 0.02,
            }
        ]
    )
    existing.to_csv(tmp_path / "shadow_challenger.csv", index=False)
    proposed = pd.DataFrame(
        [
            {
                "candidate": "candidate",
                "status": "prospective_near_miss_shadow",
                "retrospective_focus_cagr_delta_vs_native": 0.03,
                "all_execution_lags_positive_vs_native": True,
                "activation_episode_count": 18,
            }
        ]
    )

    frozen = severity.freeze_severity_shadow_challenger(tmp_path, proposed)

    assert frozen.iloc[0]["status"] == "prospective_observation_only"
    assert frozen.iloc[0]["retrospective_focus_cagr_delta_vs_native"] == pytest.approx(0.02)
    assert bool(frozen.iloc[0]["all_execution_lags_positive_vs_native"])
    assert frozen.iloc[0]["activation_episode_count"] == 18
