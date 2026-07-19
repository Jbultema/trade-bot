from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.cycle_tracker import (
    PHASES,
    build_cycle_candidate_scores,
    build_cycle_crisis_playback,
    build_cycle_feature_snapshot,
    build_cycle_path_state_history,
    build_cycle_validation_observations,
    build_path_candidate_validation_observations,
    build_path_transition_forecast,
    build_path_validation_observations,
    build_phase_candidate_frontier,
    run_cycle_tracker,
    summarize_path_phase_reliability,
    summarize_phase_reliability,
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
    assert not result.path_validation_metrics.empty
    assert "0m" in set(result.phase_reliability["horizon"].astype(str))
    assert "0m" in set(result.path_reliability["horizon"].astype(str))
    assert (tmp_path / "cycle_phase_probabilities.csv").exists()
    assert (tmp_path / "cycle_transition_forecast.csv").exists()
    assert (tmp_path / "cycle_candidate_scores.csv").exists()
    assert (tmp_path / "cycle_phase_candidate_frontier.csv").exists()
    assert (tmp_path / "cycle_phase_reliability.csv").exists()
    assert (tmp_path / "cycle_path_validation_metrics.csv").exists()
    assert (tmp_path / "cycle_path_state_history.csv").exists()
    assert (tmp_path / "cycle_path_transition_forecast.csv").exists()
    assert (tmp_path / "cycle_path_reliability.csv").exists()
    assert (tmp_path / "cycle_crisis_playback.csv").exists()
    assert (tmp_path / "summary.md").exists()


def test_path_state_history_adds_memory_and_duration() -> None:
    prices = _cycle_prices("unwind", periods=420)

    path = build_cycle_path_state_history(
        prices,
        min_train_days=252,
        state_step_days=21,
    )

    assert not path.empty
    assert {
        "evidence_phase",
        "path_phase",
        "phase_duration_days",
        "phase_duration_bucket",
        "prior_unwind_seen_504d",
        "prior_bottoming_seen_504d",
        "transition_reason",
    }.issubset(path.columns)
    assert pd.to_numeric(path["phase_duration_days"], errors="coerce").ge(0).all()
    assert path["transition_allowed"].astype(bool).all()
    assert set(path["path_phase"]).intersection({"early_unwind", "liquidation"})


def test_path_state_history_breaks_out_of_stale_normal_during_deep_stress() -> None:
    prices = _cycle_prices("unwind", periods=520)

    path = build_cycle_path_state_history(
        prices,
        min_train_days=252,
        state_step_days=21,
    )

    deep_stress = path[
        pd.to_numeric(path["qqq_drawdown_252d"], errors="coerce").le(-0.18)
    ]
    assert not deep_stress.empty
    assert set(deep_stress["path_phase"]).intersection({"early_unwind", "liquidation"})


def test_path_transition_constrains_post_unwind_without_prior_unwind() -> None:
    phase_probabilities = pd.DataFrame(
        [
            {
                "phase": "post_unwind_compounding",
                "probability": 0.70,
                "dominant_phase": "post_unwind_compounding",
                "horizon": "0m",
                "horizon_days": 0,
                "source": "test",
            },
            {
                "phase": "normal_cycle",
                "probability": 0.30,
                "dominant_phase": "post_unwind_compounding",
                "horizon": "0m",
                "horizon_days": 0,
                "source": "test",
            },
        ]
    )
    path_history = pd.DataFrame(
        [
            {
                "as_of_date": "2026-07-16",
                "path_phase": "normal_cycle",
                "path_probability": 0.80,
                "phase_duration_days": 90,
                "prior_unwind_seen_504d": False,
                "prior_bottoming_seen_504d": False,
                "qqq_drawdown_252d": -0.02,
                "spy_drawdown_252d": -0.01,
                "normal_cycle_probability": 0.80,
                "post_unwind_compounding_probability": 0.20,
            }
        ]
    )

    forecast = build_path_transition_forecast(
        phase_probabilities,
        path_history,
        scenario_lattice=None,
        horizons=(0, 21),
    )

    one_month = forecast[forecast["horizon"].astype(str).eq("1m")]
    post = one_month[one_month["phase"].astype(str).eq("post_unwind_compounding")].iloc[0]
    normal = one_month[one_month["phase"].astype(str).eq("normal_cycle")].iloc[0]
    assert float(post["probability"]) < float(normal["probability"])
    assert "needs unwind/recovery path" in str(post["precondition"])


def test_path_reliability_summarizes_path_fit() -> None:
    observations = pd.DataFrame(
        [
            {
                "path_phase": "liquidation",
                "path_probability": 0.80,
                "phase_duration_days": 21,
                "horizon": "1m",
                "horizon_days": 21,
                "qqq_forward_return": -0.10,
                "spy_forward_return": -0.08,
                "bil_forward_return": 0.001,
                "qqq_forward_drawdown": -0.15,
                "path_phase_fit": True,
            },
            {
                "path_phase": "liquidation",
                "path_probability": 0.65,
                "phase_duration_days": 42,
                "horizon": "1m",
                "horizon_days": 21,
                "qqq_forward_return": 0.05,
                "spy_forward_return": 0.03,
                "bil_forward_return": 0.001,
                "qqq_forward_drawdown": -0.03,
                "path_phase_fit": False,
            },
        ]
    )

    reliability = summarize_path_phase_reliability(observations)

    assert not reliability.empty
    row = reliability.iloc[0]
    assert row["path_phase"] == "liquidation"
    assert row["path_fit_rate"] == pytest.approx(0.5)


def test_path_reliability_includes_nowcast_agreement() -> None:
    prices = _cycle_prices("acceleration", periods=340)
    origin = str(prices.index[260].date())
    path_history = pd.DataFrame(
        [
            {
                "as_of_date": origin,
                "path_phase": "post_unwind_compounding",
                "evidence_phase": "post_unwind_compounding",
                "path_probability": 0.72,
                "phase_duration_days": 63,
            },
            {
                "as_of_date": origin,
                "path_phase": "post_unwind_compounding",
                "evidence_phase": "acceleration",
                "path_probability": 0.64,
                "phase_duration_days": 42,
            },
        ]
    )

    observations = build_path_validation_observations(
        prices,
        path_history,
        horizons=(0, 21),
    )
    nowcast = observations[observations["horizon"].astype(str).eq("0m")]

    assert len(nowcast) == 2
    assert nowcast["path_phase_fit"].tolist() == [True, False]
    assert nowcast["qqq_forward_return"].isna().all()
    assert (pd.to_datetime(nowcast["entry_date"]) == pd.to_datetime(nowcast["origin_date"])).all()

    reliability = summarize_path_phase_reliability(observations)
    nowcast_summary = reliability[reliability["horizon"].astype(str).eq("0m")].iloc[0]
    assert nowcast_summary["path_fit_rate"] == pytest.approx(0.5)
    assert "no forward realized outcome" in str(nowcast_summary["expected_behavior"])


def test_path_candidate_validation_uses_decoded_path_phase() -> None:
    prices = _cycle_prices("unwind", periods=420)
    path_history = pd.DataFrame(
        [
            {
                "as_of_date": str(prices.index[260].date()),
                "path_phase": "liquidation",
                "path_probability": 0.82,
                "phase_duration_days": 21,
            }
        ]
    )

    observations = build_path_candidate_validation_observations(
        prices,
        path_history,
        tickers=("QQQ", "BIL"),
        horizons=(21,),
    )

    assert not observations.empty
    assert set(observations["dominant_phase"]) == {"liquidation"}
    assert (
        pd.to_datetime(observations["entry_date"])
        > pd.to_datetime(observations["origin_date"])
    ).all()


def test_phase_reliability_summarizes_classifier_fit() -> None:
    observations = pd.DataFrame(
        [
            {
                "origin_date": "2020-01-01",
                "dominant_phase": "acceleration",
                "phase_probability": 0.70,
                "horizon": "1m",
                "horizon_days": 21,
                "ticker": "QQQ",
                "forward_return": 0.08,
                "forward_max_drawdown": -0.02,
                "spy_forward_return": 0.04,
                "qqq_forward_return": 0.08,
                "bil_forward_return": 0.001,
            },
            {
                "origin_date": "2020-02-01",
                "dominant_phase": "acceleration",
                "phase_probability": 0.65,
                "horizon": "1m",
                "horizon_days": 21,
                "ticker": "QQQ",
                "forward_return": -0.04,
                "forward_max_drawdown": -0.09,
                "spy_forward_return": -0.02,
                "qqq_forward_return": -0.04,
                "bil_forward_return": 0.001,
            },
        ]
    )

    reliability = summarize_phase_reliability(observations)

    assert not reliability.empty
    row = reliability.iloc[0]
    assert row["dominant_phase"] == "acceleration"
    assert row["phase_fit_rate"] == pytest.approx(0.5)
    assert row["expected_behavior"]


def test_crisis_playback_replays_phase_probabilities() -> None:
    prices = _cycle_prices("unwind", periods=900, start="2018-01-01")

    playback = build_cycle_crisis_playback(prices, horizons=(0, 21), origin_step_days=21)

    assert not playback.empty
    assert {"crisis", "stage", "phase", "phase_probability", "phase_fit"}.issubset(
        playback.columns
    )
    assert set(playback["horizon"]) == {"0m", "1m"}
    nowcast = playback[playback["horizon"].astype(str).eq("0m")]
    assert nowcast["qqq_forward_return"].isna().all()
    assert nowcast["phase_fit"].astype(bool).all()


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
    assert {"origin_confidence", "evidence_quality", "evidence_flags", "exposure_family"}.issubset(
        frontier.columns
    )


def test_phase_candidate_frontier_caps_one_off_ai_stress_winner() -> None:
    prices = _cycle_prices("acceleration", periods=420)
    transition_forecast = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "horizon_days": 21,
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
                "horizon": "1m",
                "horizon_days": 21,
                "ticker": "QQQ",
                "asset_role": "ai_growth",
                "origins": 1,
                "median_forward_return": 0.40,
                "median_forward_drawdown": -0.05,
                "median_excess_vs_spy": 0.25,
                "median_excess_vs_qqq": 0.30,
                "hit_rate_vs_qqq": 1.0,
                "severe_drawdown_rate": 0.0,
                "phase_rank_score": 0.50,
            },
            {
                "dominant_phase": "liquidation",
                "horizon": "1m",
                "horizon_days": 21,
                "ticker": "BIL",
                "asset_role": "cash_defensive",
                "origins": 24,
                "median_forward_return": 0.01,
                "median_forward_drawdown": -0.001,
                "median_excess_vs_spy": 0.05,
                "median_excess_vs_qqq": 0.08,
                "hit_rate_vs_qqq": 0.83,
                "severe_drawdown_rate": 0.0,
                "phase_rank_score": 0.20,
            },
        ]
    )

    frontier = build_phase_candidate_frontier(
        prices,
        pd.DataFrame(),
        transition_forecast,
        validation_metrics,
        tickers=("QQQ", "BIL"),
    )

    assert frontier.iloc[0]["ticker"] == "BIL"
    qqq = frontier[frontier["ticker"].eq("QQQ")].iloc[0]
    assert qqq["frontier_role"] in {"thin_sample_watch", "avoid"}
    assert qqq["evidence_quality"] == "one_off_sample"
    assert "ai_fragility_conflict" in qqq["evidence_flags"]
    assert qqq["origin_confidence"] < 0.10


def test_phase_candidate_frontier_limits_redundant_exposure_families() -> None:
    prices = _cycle_prices("acceleration", periods=420)
    prices["VOO"] = prices["SPY"] * 1.001
    prices["IVV"] = prices["SPY"] * 0.999
    prices["SPLG"] = prices["SPY"] * 1.002
    transition_forecast = pd.DataFrame(
        [
            {
                "horizon": "3m",
                "horizon_days": 63,
                "phase": "normal_cycle",
                "probability": 0.40,
                "dominant_phase": "normal_cycle",
                "source": "test",
            }
        ]
    )
    validation_metrics = pd.DataFrame(
        [
            {
                "dominant_phase": "normal_cycle",
                "horizon": "3m",
                "horizon_days": 63,
                "ticker": ticker,
                "asset_role": "broad_equity",
                "origins": 30,
                "median_forward_return": 0.04,
                "median_forward_drawdown": -0.04,
                "median_excess_vs_spy": 0.0,
                "median_excess_vs_qqq": -0.02,
                "hit_rate_vs_qqq": 0.45,
                "severe_drawdown_rate": 0.0,
                "phase_rank_score": rank_score,
            }
            for ticker, rank_score in [
                ("SPY", 0.35),
                ("VOO", 0.34),
                ("IVV", 0.33),
                ("SPLG", 0.32),
                ("RSP", 0.25),
                ("QQQ", 0.20),
                ("SMH", 0.19),
            ]
        ]
    )

    frontier = build_phase_candidate_frontier(
        prices,
        pd.DataFrame(),
        transition_forecast,
        validation_metrics,
        tickers=("SPY", "VOO", "IVV", "SPLG", "RSP", "QQQ", "SMH"),
        top_n_per_phase=4,
    )

    assert len(frontier) == 4
    assert frontier["exposure_family"].tolist().count("us_large_cap_beta") == 1
    assert "RSP" in set(frontier["ticker"])


def test_liquidation_frontier_reentry_role_requires_longer_horizon() -> None:
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
                "horizon": "1m",
                "horizon_days": 21,
                "phase": "liquidation",
                "probability": 0.65,
                "dominant_phase": "liquidation",
                "source": "test",
            },
            {
                "horizon": "1y",
                "horizon_days": 252,
                "phase": "liquidation",
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
                "horizon": horizon,
                "horizon_days": horizon_days,
                "ticker": "QQQ",
                "asset_role": "ai_growth",
                "origins": 12,
                "median_forward_return": 0.30,
                "median_forward_drawdown": -0.08,
                "median_excess_vs_spy": 0.10,
                "median_excess_vs_qqq": 0.20,
                "hit_rate_vs_qqq": 0.90,
                "severe_drawdown_rate": 0.0,
                "phase_rank_score": 0.50,
            }
            for horizon, horizon_days in [("1m", 21), ("1y", 252)]
        ]
    )

    frontier = build_phase_candidate_frontier(
        prices,
        phase_probabilities,
        transition_forecast,
        validation_metrics,
        tickers=("QQQ",),
    )

    short_role = frontier[frontier["horizon_days"].eq(21)].iloc[0]["frontier_role"]
    long_role = frontier[frontier["horizon_days"].eq(252)].iloc[0]["frontier_role"]
    assert short_role in {"watch", "avoid"}
    assert long_role in {"scale_reentry", "reentry_watch", "watch", "avoid"}


def test_phase_candidate_frontier_penalizes_current_ai_break_theme() -> None:
    prices = _cycle_prices("unwind", periods=420)
    prices["CRWD"] = prices["QQQ"] * 1.01
    transition_forecast = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "horizon_days": 21,
                "phase": "early_unwind",
                "probability": 0.65,
                "dominant_phase": "early_unwind",
                "source": "test",
            }
        ]
    )
    validation_metrics = pd.DataFrame(
        [
            {
                "dominant_phase": "early_unwind",
                "horizon": "1m",
                "horizon_days": 21,
                "ticker": "CRWD",
                "asset_role": "ai_growth",
                "origins": 24,
                "median_forward_return": 0.25,
                "median_forward_drawdown": -0.08,
                "median_excess_vs_spy": 0.14,
                "median_excess_vs_qqq": 0.20,
                "hit_rate_vs_qqq": 0.88,
                "severe_drawdown_rate": 0.0,
                "phase_rank_score": 0.45,
            },
            {
                "dominant_phase": "early_unwind",
                "horizon": "1m",
                "horizon_days": 21,
                "ticker": "BIL",
                "asset_role": "cash_defensive",
                "origins": 24,
                "median_forward_return": 0.01,
                "median_forward_drawdown": -0.001,
                "median_excess_vs_spy": 0.05,
                "median_excess_vs_qqq": 0.08,
                "hit_rate_vs_qqq": 0.80,
                "severe_drawdown_rate": 0.0,
                "phase_rank_score": 0.22,
            },
        ]
    )

    frontier = build_phase_candidate_frontier(
        prices,
        pd.DataFrame(),
        transition_forecast,
        validation_metrics,
        tickers=("CRWD", "BIL"),
    )

    assert frontier.iloc[0]["ticker"] == "BIL"
    crwd = frontier[frontier["ticker"].eq("CRWD")].iloc[0]
    assert crwd["theme_fragility_penalty"] > 0.0
    assert "ai_fragility_conflict" in crwd["evidence_flags"]
    assert "cycle_leader_unwind_risk" in crwd["evidence_flags"]
    assert crwd["frontier_role"] in {"watch", "avoid"}


def test_phase_candidate_frontier_penalizes_ubiquitous_winners() -> None:
    prices = _cycle_prices("acceleration", periods=420)
    prices["CRWD"] = prices["QQQ"] * 1.01
    transition_forecast = pd.DataFrame(
        [
            {
                "horizon": "3m",
                "horizon_days": 63,
                "phase": "pre_break",
                "probability": 0.55,
                "dominant_phase": "pre_break",
                "source": "test",
            }
        ]
    )
    validation_rows = [
        {
            "dominant_phase": phase,
            "horizon": "3m",
            "horizon_days": 63,
            "ticker": "CRWD",
            "asset_role": "ai_growth",
            "origins": 24,
            "median_forward_return": 0.20,
            "median_forward_drawdown": -0.08,
            "median_excess_vs_spy": 0.10,
            "median_excess_vs_qqq": 0.12,
            "hit_rate_vs_qqq": 0.75,
            "severe_drawdown_rate": 0.0,
            "phase_rank_score": 0.45,
        }
        for phase in PHASES
    ]
    validation_rows.append(
        {
            "dominant_phase": "pre_break",
            "horizon": "3m",
            "horizon_days": 63,
            "ticker": "BIL",
            "asset_role": "cash_defensive",
            "origins": 24,
            "median_forward_return": 0.01,
            "median_forward_drawdown": -0.001,
            "median_excess_vs_spy": 0.04,
            "median_excess_vs_qqq": 0.07,
            "hit_rate_vs_qqq": 0.72,
            "severe_drawdown_rate": 0.0,
            "phase_rank_score": 0.45,
        }
    )

    frontier = build_phase_candidate_frontier(
        prices,
        pd.DataFrame(),
        transition_forecast,
        pd.DataFrame(validation_rows),
        tickers=("CRWD", "BIL"),
    )

    assert frontier.iloc[0]["ticker"] == "BIL"
    crwd = frontier[frontier["ticker"].eq("CRWD")].iloc[0]
    assert crwd["ubiquity_penalty"] > 0.0
    assert "weak_phase_specificity" in crwd["evidence_flags"]
    assert "ubiquitous_winner" in crwd["evidence_flags"]


def _cycle_prices(kind: str, *, periods: int = 320, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=periods)
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
