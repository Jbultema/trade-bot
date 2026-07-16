from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.cycle_tracker import (
    build_cycle_candidate_scores,
    build_cycle_feature_snapshot,
    build_cycle_validation_observations,
    build_phase_candidate_frontier,
    run_cycle_tracker,
)


def test_cycle_feature_probabilities_sum_to_one() -> None:
    prices = _cycle_prices("acceleration")

    feature = build_cycle_feature_snapshot(prices)

    probabilities = feature["probabilities"]
    assert isinstance(probabilities, dict)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert feature["dominant_phase"] in probabilities


def test_unwind_prices_raise_unwind_or_liquidation_probability() -> None:
    calm = build_cycle_feature_snapshot(_cycle_prices("acceleration"))
    unwind = build_cycle_feature_snapshot(_cycle_prices("unwind"))

    calm_probs = calm["probabilities"]
    unwind_probs = unwind["probabilities"]
    assert isinstance(calm_probs, dict)
    assert isinstance(unwind_probs, dict)

    unwind_pressure = unwind_probs["early_unwind"] + unwind_probs["liquidation"]
    calm_pressure = calm_probs["early_unwind"] + calm_probs["liquidation"]
    assert unwind_pressure > calm_pressure


def test_cycle_validation_uses_next_session_after_origin() -> None:
    prices = _cycle_prices("acceleration", periods=360)

    observations = build_cycle_validation_observations(
        prices,
        tickers=("QQQ", "BIL"),
        horizons=(21,),
        min_train_days=252,
        origin_step_days=40,
    )

    assert not observations.empty
    assert (
        pd.to_datetime(observations["entry_date"])
        > pd.to_datetime(observations["origin_date"])
    ).all()
    assert (
        pd.to_datetime(observations["end_date"])
        > pd.to_datetime(observations["entry_date"])
    ).all()


def test_cycle_tracker_writes_artifacts(tmp_path) -> None:
    prices = _cycle_prices("unwind", periods=420)

    result = run_cycle_tracker(
        prices=prices,
        output_dir=tmp_path,
        candidate_tickers=("SPY", "QQQ", "BIL"),
        horizons=(0, 21, 63),
        min_train_days=252,
        origin_step_days=63,
    )

    assert result.phase_probabilities["probability"].sum() == pytest.approx(1.0)
    assert set(result.phase_probabilities["horizon"]) == {"0m"}
    assert not result.transition_forecast.empty
    assert "0m" in set(result.transition_forecast["horizon"])
    assert 0 not in set(result.validation_observations["horizon_days"])
    assert not result.candidate_scores.empty
    assert not result.phase_candidate_frontier.empty
    assert (tmp_path / "cycle_phase_probabilities.csv").exists()
    assert (tmp_path / "cycle_transition_forecast.csv").exists()
    assert (tmp_path / "cycle_candidate_scores.csv").exists()
    assert (tmp_path / "cycle_phase_candidate_frontier.csv").exists()
    assert (tmp_path / "summary.md").exists()


def test_liquidation_phase_favors_defensive_candidates() -> None:
    prices = _cycle_prices("unwind", periods=420)
    feature = build_cycle_feature_snapshot(prices)
    phase_probabilities = pd.DataFrame(
        [
            {
                "as_of_date": feature["as_of_date"],
                "horizon": "0m",
                "horizon_days": 0,
                "phase": "liquidation",
                "probability": 0.70,
                "dominant_phase": "liquidation",
                "source": "test",
            },
            {
                "as_of_date": feature["as_of_date"],
                "horizon": "0m",
                "horizon_days": 0,
                "phase": "early_unwind",
                "probability": 0.30,
                "dominant_phase": "liquidation",
                "source": "test",
            },
        ]
    )
    validation_metrics = pd.DataFrame(
        [
            {
                "dominant_phase": "liquidation",
                "horizon_days": 63,
                "ticker": "BIL",
                "phase_rank_score": 0.35,
                "median_forward_return": 0.01,
                "median_excess_vs_spy": 0.05,
                "median_excess_vs_qqq": 0.08,
                "hit_rate_vs_qqq": 0.80,
                "median_forward_drawdown": -0.001,
                "origins": 10,
            },
            {
                "dominant_phase": "liquidation",
                "horizon_days": 63,
                "ticker": "QQQ",
                "phase_rank_score": -0.20,
                "median_forward_return": -0.08,
                "median_excess_vs_spy": -0.02,
                "median_excess_vs_qqq": 0.0,
                "hit_rate_vs_qqq": 0.50,
                "median_forward_drawdown": -0.18,
                "origins": 10,
            },
        ]
    )

    scores = build_cycle_candidate_scores(
        prices,
        phase_probabilities,
        validation_metrics,
        tickers=("QQQ", "BIL"),
        horizon_days=63,
    )

    assert scores.iloc[0]["ticker"] == "BIL"
    assert scores.iloc[0]["candidate_role"] == "defend"


def test_phase_candidate_frontier_scores_phase_horizon_winners() -> None:
    prices = _cycle_prices("unwind", periods=420)
    phase_probabilities = pd.DataFrame(
        [
            {
                "as_of_date": "2026-07-16",
                "horizon": "0m",
                "horizon_days": 0,
                "phase": "liquidation",
                "probability": 0.70,
                "dominant_phase": "liquidation",
                "source": "test",
            }
        ]
    )
    transition_forecast = pd.DataFrame(
        [
            {
                "horizon": "3m",
                "horizon_days": 63,
                "phase": "liquidation",
                "probability": 0.65,
                "dominant_phase": "liquidation",
                "source": "test",
            }
        ]
    )
    validation_metrics = pd.DataFrame(
        [
            {
                "dominant_phase": "liquidation",
                "horizon": "3m",
                "horizon_days": 63,
                "ticker": "BIL",
                "asset_role": "cash_defensive",
                "origins": 12,
                "median_forward_return": 0.01,
                "median_forward_drawdown": -0.001,
                "median_excess_vs_spy": 0.05,
                "median_excess_vs_qqq": 0.08,
                "hit_rate_vs_qqq": 0.83,
                "severe_drawdown_rate": 0.0,
                "phase_rank_score": 0.40,
            },
            {
                "dominant_phase": "liquidation",
                "horizon": "3m",
                "horizon_days": 63,
                "ticker": "QQQ",
                "asset_role": "ai_growth",
                "origins": 12,
                "median_forward_return": -0.08,
                "median_forward_drawdown": -0.18,
                "median_excess_vs_spy": -0.02,
                "median_excess_vs_qqq": 0.0,
                "hit_rate_vs_qqq": 0.50,
                "severe_drawdown_rate": 0.58,
                "phase_rank_score": -0.25,
            },
        ]
    )

    frontier = build_phase_candidate_frontier(
        prices,
        phase_probabilities,
        transition_forecast,
        validation_metrics,
        tickers=("QQQ", "BIL"),
    )

    assert not frontier.empty
    assert frontier.iloc[0]["ticker"] == "BIL"
    assert frontier.iloc[0]["frontier_role"] == "defend"
    assert frontier.iloc[0]["rank"] == 1


def _cycle_prices(kind: str, *, periods: int = 320) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=periods)
    data: dict[str, list[float]] = {}
    base_returns = {
        "SPY": 0.00045,
        "QQQ": 0.00065,
        "RSP": 0.00035,
        "IWM": 0.00030,
        "SMH": 0.00090,
        "SOXX": 0.00085,
        "IGV": 0.00055,
        "HYG": 0.00015,
        "LQD": 0.00010,
        "BIL": 0.00002,
        "GLD": 0.00015,
        "TLT": 0.00005,
        "VEA": 0.00025,
        "VIXY": -0.00070,
    }
    for ticker, daily_return in base_returns.items():
        values = [100.0]
        for day in range(1, periods):
            drift = daily_return
            if kind == "unwind" and day > periods - 90:
                if ticker in {"QQQ", "SMH", "SOXX", "IGV"}:
                    drift = -0.0045
                elif ticker == "SPY":
                    drift = -0.0025
                elif ticker == "RSP":
                    drift = -0.0010
                elif ticker == "HYG":
                    drift = -0.0015
                elif ticker == "LQD":
                    drift = 0.00005
                elif ticker == "BIL":
                    drift = 0.00008
                elif ticker in {"GLD", "TLT", "VIXY"}:
                    drift = 0.0025
            values.append(values[-1] * (1.0 + drift))
        data[ticker] = values
    return pd.DataFrame(data, index=index)
