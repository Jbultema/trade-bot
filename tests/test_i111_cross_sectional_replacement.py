from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.config import ExecutionConfig
from trade_bot.research.i111_cross_sectional_replacement import (
    CLUSTERED_AI_STRESS_THRESHOLD,
    apply_clustered_stress_replacement,
    fixed_cross_sectional_replacement_specs,
    fixed_execution_profiles,
)


def test_fixed_replacement_slate_is_predeclared() -> None:
    specs = fixed_cross_sectional_replacement_specs()

    assert [spec.name for spec in specs] == [
        "native_reference",
        "defer_ai_increases_to_bil",
        "defer_ai_increases_to_rsp",
    ]
    assert CLUSTERED_AI_STRESS_THRESHOLD == 0.75


def test_clustered_stress_allows_ai_exits_but_redirects_increases() -> None:
    index = pd.bdate_range("2025-01-02", periods=3)
    weights = pd.DataFrame(
        {
            "NVDA": [0.20, 0.10, 0.30],
            "MSFT": [0.10, 0.30, 0.20],
            "BIL": [0.70, 0.60, 0.50],
        },
        index=index,
    )
    stress = pd.Series([0.0, 0.75, 0.75], index=index)

    transformed, path = apply_clustered_stress_replacement(
        weights,
        stress,
        destination="BIL",
    )

    assert transformed.loc[index[1], "NVDA"] == pytest.approx(0.10)
    assert transformed.loc[index[1], "MSFT"] == pytest.approx(0.10)
    assert transformed.loc[index[1], "BIL"] == pytest.approx(0.80)
    assert transformed.loc[index[2], "NVDA"] == pytest.approx(0.10)
    assert transformed.loc[index[2], "MSFT"] == pytest.approx(0.10)
    assert transformed.loc[index[2], "BIL"] == pytest.approx(0.80)
    assert transformed.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert path["blocked_ai_increase"].tolist() == pytest.approx([0.0, 0.20, 0.30])


def test_replacement_is_causal_and_ignores_future_weight_changes() -> None:
    index = pd.bdate_range("2025-02-03", periods=4)
    weights = pd.DataFrame(
        {
            "NVDA": [0.10, 0.30, 0.20, 0.40],
            "RSP": [0.90, 0.70, 0.80, 0.60],
        },
        index=index,
    )
    stress = pd.Series([0.0, 0.75, 0.75, 0.0], index=index)
    altered_future = weights.copy()
    altered_future.loc[index[3], :] = [0.95, 0.05]

    original, _ = apply_clustered_stress_replacement(weights, stress, destination="RSP")
    altered, _ = apply_clustered_stress_replacement(
        altered_future,
        stress,
        destination="RSP",
    )

    pd.testing.assert_frame_equal(original.iloc[:3], altered.iloc[:3])
    assert original.loc[index[3], "NVDA"] == pytest.approx(0.40)


def test_execution_profiles_match_the_fixed_adversarial_slate() -> None:
    names = [name for name, _ in fixed_execution_profiles(ExecutionConfig())]

    assert names == [
        "wednesday_lag1",
        "monday_lag1",
        "tuesday_lag1",
        "thursday_lag1",
        "friday_lag1",
        "daily_lag1",
        "wednesday_lag2",
        "wednesday_lag5",
    ]
