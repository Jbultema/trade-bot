from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_bot.research.native_timing_hazard import (
    add_episode_cluster_weights,
    apply_continuous_defense_budget,
    build_forward_break_labels,
    build_market_hazard_panel,
    continuous_defense_target,
    freeze_shadow_candidate,
    policy_parameter_grid,
)


def _prices(rows: int = 700) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=rows)
    trend = np.linspace(100.0, 180.0, rows)
    return pd.DataFrame(
        {
            "SPY": trend,
            "QQQ": trend * 1.05,
            "RSP": trend * 0.98,
            "HYG": np.linspace(80.0, 90.0, rows),
            "LQD": np.linspace(110.0, 115.0, rows),
            "VIXY": np.linspace(40.0, 20.0, rows),
            "TLT": np.linspace(130.0, 105.0, rows),
            "SHY": np.linspace(82.0, 85.0, rows),
            "SMH": trend * 1.10,
            "IGV": trend * 1.02,
            "XLF": trend * 0.92,
            "XLI": trend * 0.95,
            "XLV": trend * 0.90,
            "GLD": np.linspace(150.0, 170.0, rows),
        },
        index=index,
    )


def test_forward_break_labels_leave_unmatured_tail_missing() -> None:
    prices = _prices(100)
    origins = pd.Series([prices.index[10], prices.index[90]])

    labels = build_forward_break_labels(
        prices,
        origins,
        horizon_sessions=20,
        break_threshold=-0.10,
    )

    assert labels.loc[0, "maturity_date"] == prices.index[30]
    assert labels.loc[0, "forward_break"] == 0.0
    assert pd.isna(labels.loc[1, "maturity_date"])
    assert pd.isna(labels.loc[1, "forward_break"])


def test_market_panel_features_use_prior_close() -> None:
    prices = _prices()
    panel = build_market_hazard_panel(prices, min_history_sessions=504)
    origin = pd.Timestamp(panel.iloc[4]["origin_date"])
    prior_position = prices.index.get_loc(origin) - 1
    expected = prices["SPY"].iloc[prior_position] / prices["SPY"].iloc[
        prior_position - 21
    ] - 1.0

    assert panel.iloc[4]["spy_return_21"] == pytest.approx(expected)


def test_episode_weights_equalize_clusters() -> None:
    panel = pd.DataFrame(
        {
            "origin_date": pd.date_range("2024-01-01", periods=8, freq="W"),
            "forward_break": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, np.nan],
        }
    )
    weighted = add_episode_cluster_weights(panel)
    matured = weighted[weighted["forward_break"].notna()]
    cluster_weight = matured.groupby("episode_cluster")["episode_weight"].sum()

    assert cluster_weight.max() == pytest.approx(cluster_weight.min())
    assert weighted.loc[7, "episode_weight"] != weighted.loc[7, "episode_weight"]


def test_continuous_target_is_monotonic_and_age_decay_is_bounded() -> None:
    index = pd.date_range("2025-01-01", periods=3)
    parameters = {
        "defense_floor": 0.10,
        "hazard_slope": 1.20,
        "defense_ceiling": 0.90,
        "native_blend": 0.50,
        "break_acceleration": 0.10,
        "age_decay": 0.02,
    }
    base = pd.Series(0.50, index=index)
    probability = pd.Series([0.10, 0.20, 0.30], index=index)
    target = continuous_defense_target(
        probability,
        base,
        pd.Series(0.0, index=index),
        pd.Series(0.0, index=index),
        policy="family_confirm_age_existing",
        parameters=parameters,
    )
    stale = continuous_defense_target(
        probability,
        base,
        pd.Series(0.0, index=index),
        pd.Series(100.0, index=index),
        policy="family_confirm_age_existing",
        parameters=parameters,
    )

    assert target.is_monotonic_increasing
    assert ((target - stale) <= 0.075 + 1e-12).all()
    assert (stale >= parameters["defense_floor"]).all()


def test_continuous_budget_preserves_valid_weights_and_spy_bridge() -> None:
    index = pd.date_range("2025-01-01", periods=2)
    base = pd.DataFrame(
        {"QQQ": [0.30, 0.00], "BIL": [0.70, 1.00]},
        index=index,
    )
    target = pd.Series([0.40, 0.50], index=index)

    adjusted = apply_continuous_defense_budget(
        base,
        target,
        spy_bridge=True,
    )

    assert np.allclose(adjusted.sum(axis=1), 1.0)
    assert (adjusted >= 0.0).all().all()
    assert (adjusted.drop(columns="BIL") <= 0.35 + 1e-12).all().all()
    assert adjusted.loc[index[1], "SPY"] == pytest.approx(0.35)
    assert adjusted.loc[index[1], "BIL"] == pytest.approx(0.65)


def test_policy_grid_is_architecturally_distinct() -> None:
    plain = policy_parameter_grid("family_continuous_existing")
    full = policy_parameter_grid("family_confirm_age_existing")

    assert len(plain) == 24
    assert len(full) == 96
    assert {row["break_acceleration"] for row in plain} == {0.0}
    assert {row["age_decay"] for row in plain} == {0.0}
    assert {row["break_acceleration"] for row in full} == {0.05, 0.10}
    assert {row["age_decay"] for row in full} == {0.01, 0.02}


def test_shadow_candidate_freeze_does_not_roll_forward(tmp_path) -> None:
    existing = pd.DataFrame(
        [
            {
                "candidate": "candidate_v1",
                "status": "prospective_shadow_frozen",
                "shadow_start_after_market_date": "2026-07-23",
            }
        ]
    )
    existing.to_csv(tmp_path / "shadow_candidate.csv", index=False)
    proposed = existing.copy()
    proposed["shadow_start_after_market_date"] = "2026-08-01"

    frozen = freeze_shadow_candidate(tmp_path, proposed)

    assert frozen.iloc[0]["shadow_start_after_market_date"] == "2026-07-23"
