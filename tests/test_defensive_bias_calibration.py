from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.defensive_bias_calibration import (
    bounded_sleeve_shift,
    build_adjusted_weight_path,
    counterfactual_utility_delta,
    defensive_weight,
    hierarchical_posterior,
)


def test_bounded_sleeve_shift_preserves_cap_and_existing_risk_mix() -> None:
    weights = pd.Series({"QQQ": 0.20, "SPY": 0.10, "BIL": 0.70})

    shifted, amount = bounded_sleeve_shift(weights, "defense_relief", cap=0.05)

    assert amount == pytest.approx(0.05)
    assert shifted.sum() == pytest.approx(1.0)
    assert shifted["BIL"] == pytest.approx(0.65)
    assert shifted["QQQ"] / shifted["SPY"] == pytest.approx(2.0)


def test_defense_relief_does_not_invent_risk_asset() -> None:
    shifted, amount = bounded_sleeve_shift(
        pd.Series({"BIL": 1.0}),
        "defense_relief",
        cap=0.05,
    )

    assert amount == 0.0
    assert shifted.to_dict() == {"BIL": 1.0}


def test_defensive_weight_materializes_unallocated_residual() -> None:
    assert defensive_weight(pd.Series({"QQQ": 0.25, "BIL": 0.50})) == pytest.approx(0.75)


def test_relief_preserves_residual_cash_semantics_instead_of_creating_bil() -> None:
    weights = pd.Series({"QQQ": 0.25, "BIL": 0.02})

    shifted, amount = bounded_sleeve_shift(weights, "defense_relief", cap=0.05)

    assert amount == pytest.approx(0.05)
    assert shifted["QQQ"] == pytest.approx(0.30)
    assert shifted["BIL"] == pytest.approx(0.0)
    assert shifted.sum() == pytest.approx(0.30)
    assert defensive_weight(shifted) == pytest.approx(0.70)


def test_counterfactual_utility_penalizes_drawdown_damage_more_than_improvement() -> None:
    improvement = counterfactual_utility_delta(0.0, 0.01)
    deterioration = counterfactual_utility_delta(0.0, -0.01)

    assert improvement == pytest.approx(0.0075)
    assert deterioration == pytest.approx(-0.015)


def test_hierarchical_posterior_uses_only_matured_rows_and_counts_origins_once() -> None:
    rows = []
    for origin_number in range(30):
        origin = pd.Timestamp("2018-01-31") + pd.offsets.MonthEnd(origin_number)
        for strategy in ("i111_a", "i111_b", "i111_c"):
            rows.append(
                {
                    "origin_date": origin,
                    "maturity_date": origin + pd.Timedelta(days=25),
                    "strategy": strategy,
                    "family": "i111",
                    "action": "defense_relief",
                    "utility_delta": 0.01,
                }
            )
    # This attractive future outcome must not enter the point-in-time estimate.
    rows.append(
        {
            "origin_date": pd.Timestamp("2025-01-31"),
            "maturity_date": pd.Timestamp("2025-03-01"),
            "strategy": "i111_a",
            "family": "i111",
            "action": "defense_relief",
            "utility_delta": 1.0,
        }
    )
    history = pd.DataFrame(rows)

    posterior = hierarchical_posterior(
        history,
        strategy="i111_a",
        family="i111",
        action="defense_relief",
        origin=pd.Timestamp("2021-01-31"),
    )

    assert posterior["global_origins"] == 30
    assert posterior["family_origins"] == 30
    assert posterior["strategy_observations"] == 30
    assert 0.0 < posterior["strategy_posterior_mean"] < 0.01
    assert posterior["hierarchical_action_eligible"] is True


def test_confirmation_gate_blocks_relief_but_not_risk_restraint() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="D")
    origins = pd.DataFrame(
        [
            {
                "origin_date": dates[0],
                "defense_relief_allowed": False,
                "risk_timing_state": "confirmed_break",
            }
        ]
    )
    estimates = pd.DataFrame(
        [
            {
                "origin_date": dates[0],
                "action": "defense_relief",
                "action_allowed": True,
            },
            {
                "origin_date": dates[0],
                "action": "risk_restraint",
                "action_allowed": True,
            },
        ]
    )
    defensive = pd.DataFrame(
        {"QQQ": 0.30, "BIL": 0.70},
        index=dates,
    )
    risk_on = pd.DataFrame(
        {"QQQ": 0.90, "BIL": 0.10},
        index=dates,
    )

    blocked = build_adjusted_weight_path(
        defensive,
        origins,
        estimates,
        policy="hierarchical_confirmation_symmetric_5pp",
        defense_threshold=0.60,
        low_defense_threshold=0.20,
        shift_cap=0.05,
        defensive_ticker="BIL",
    )
    restrained = build_adjusted_weight_path(
        risk_on,
        origins,
        estimates,
        policy="hierarchical_confirmation_symmetric_5pp",
        defense_threshold=0.60,
        low_defense_threshold=0.20,
        shift_cap=0.05,
        defensive_ticker="BIL",
    )

    pd.testing.assert_frame_equal(blocked, defensive)
    assert restrained["BIL"].tolist() == pytest.approx([0.15] * len(dates))
    assert restrained["QQQ"].tolist() == pytest.approx([0.85] * len(dates))
