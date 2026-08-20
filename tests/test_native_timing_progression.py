from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.native_timing_progression import (
    _policy_target,
    add_warning_episode_weights,
)


def test_warning_episode_weights_equalize_eligible_runs() -> None:
    panel = pd.DataFrame(
        {
            "family": ["i111"] * 7,
            "origin_date": pd.date_range("2025-01-01", periods=7, freq="W-WED"),
            "warning_eligible": [False, True, True, True, False, True, True],
            "forward_break": [0.0] * 7,
        }
    )

    result = add_warning_episode_weights(panel)
    eligible = result[result["warning_eligible"]]
    episode_weight = eligible.groupby("warning_episode")["warning_episode_weight"].sum()

    assert len(episode_weight) == 2
    assert episode_weight.iloc[0] == pytest.approx(1.0)
    assert episode_weight.iloc[1] == pytest.approx(1.0)
    assert (result.loc[~result["warning_eligible"], "warning_episode"] == "").all()


def test_progression_restoration_only_adds_defense_above_mild_curve() -> None:
    index = pd.date_range("2025-01-01", periods=3)
    common = {
        "base_defense": pd.Series(0.60, index=index),
        "probability": pd.Series(0.20, index=index),
        "break_count": pd.Series([0.0, 1.0, 2.0], index=index),
        "non_deterioration_age": pd.Series(0.0, index=index),
    }
    mild = _policy_target(
        policy="mild_frozen",
        excess_probability=pd.Series(0.0, index=index),
        **common,
    )
    restored = _policy_target(
        policy="progression_restore_20",
        excess_probability=pd.Series([0.0, 0.5, 1.0], index=index),
        **common,
    )

    assert (restored >= mild).all()
    assert restored.iloc[0] == pytest.approx(mild.iloc[0])
    assert restored.iloc[2] - mild.iloc[2] == pytest.approx(0.20)


def test_native_severe_veto_restores_native_only_after_two_confirmations() -> None:
    index = pd.date_range("2025-01-01", periods=2)
    base = pd.Series([0.70, 0.70], index=index)
    target = _policy_target(
        policy="native_severe_veto",
        base_defense=base,
        probability=pd.Series(0.20, index=index),
        excess_probability=pd.Series(0.0, index=index),
        break_count=pd.Series([1.0, 2.0], index=index),
        non_deterioration_age=pd.Series(0.0, index=index),
    )

    assert target.iloc[0] < base.iloc[0]
    assert target.iloc[1] == pytest.approx(base.iloc[1])
