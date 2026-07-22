from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.forward_simulation import (
    REGIME_BUCKETS,
    ForwardSimulationConfig,
    ForwardSimulationValidationConfig,
    build_regime_return_library,
    rolling_origin_simulation_backtest,
    rolling_origin_strategy_rank_validation,
    scenario_bucket_probabilities,
    simulate_factor_conditioned_paths,
    simulate_regime_conditioned_paths,
    summarize_forward_simulation,
    summarize_simulation_validation,
    summarize_simulation_validation_by_horizon,
    summarize_strategy_rank_validation,
)


def test_scenario_bucket_probabilities_normalize_current_rollup() -> None:
    scenario_outlook = pd.DataFrame(
        [
            {"risk_bucket": "risk_off_then_relief", "probability": 0.20},
            {"risk_bucket": "transition", "probability": 0.25},
            {"risk_bucket": "risk_on_fragile", "probability": 0.15},
            {"risk_bucket": "risk_on", "probability": 0.40},
        ]
    )

    probabilities = scenario_bucket_probabilities(scenario_outlook)

    assert set(probabilities.index) == set(REGIME_BUCKETS)
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert probabilities["risk_off"] == pytest.approx(0.20)
    assert probabilities["transition"] == pytest.approx(0.25)
    assert probabilities["risk_on_fragile"] == pytest.approx(0.15)
    assert probabilities["risk_on"] == pytest.approx(0.40)


def test_regime_return_library_labels_known_buckets() -> None:
    returns = pd.Series(
        [0.002] * 70 + [-0.025] * 15 + [0.004] * 70 + [-0.012, 0.018] * 25 + [0.001] * 60
    )
    config = ForwardSimulationConfig(min_regime_observations=1)

    library = build_regime_return_library(returns, config=config)

    assert not library.empty
    assert set(library["regime"]).issubset(set(REGIME_BUCKETS))
    assert {"risk_off", "risk_on"}.issubset(set(library["regime"]))


def test_simulate_regime_conditioned_paths_returns_distribution() -> None:
    returns = pd.Series(
        [0.003] * 80 + [-0.02] * 20 + [0.004] * 80 + [-0.01, 0.02] * 30 + [0.001] * 60
    )
    scenario_outlook = pd.DataFrame(
        [
            {"risk_bucket": "risk_off", "probability": 0.30},
            {"risk_bucket": "transition", "probability": 0.35},
            {"risk_bucket": "risk_on_fragile", "probability": 0.10},
            {"risk_bucket": "risk_on", "probability": 0.25},
        ]
    )
    config = ForwardSimulationConfig(
        horizon_years=1,
        trading_days_per_year=20,
        paths=25,
        block_days=5,
        starting_account_value=100.0,
        annual_contribution=10.0,
        random_seed=7,
        min_regime_observations=1,
    )

    paths = simulate_regime_conditioned_paths(
        returns,
        scenario_outlook=scenario_outlook,
        config=config,
    )
    summary = summarize_forward_simulation(paths, config=config)

    assert len(paths) == 25
    assert paths["terminal_wealth"].notna().all()
    assert paths["max_drawdown"].le(0).all()
    assert summary["paths"] == 25
    assert summary["terminal_wealth_p50"] is not None
    assert summary["severe_drawdown_probability"] is not None
    assert summary["mean_regime_switches"] is not None
    assert summary["mean_covariate_match_distance"] is not None


def test_regime_return_library_adds_duration_and_covariate_state() -> None:
    index = pd.bdate_range("2025-01-02", periods=90)
    returns = pd.Series([0.002] * 35 + [-0.015] * 15 + [0.003] * 40, index=index)
    covariates = pd.DataFrame(
        {
            "credit_spread": [0.3] * 35 + [0.9] * 15 + [0.4] * 40,
            "breadth": [0.7] * 35 + [0.2] * 15 + [0.8] * 40,
        },
        index=index,
    )

    library = build_regime_return_library(
        returns,
        covariates=covariates,
        config=ForwardSimulationConfig(min_regime_observations=1),
    )

    assert "regime_duration_days" in library
    assert "cov_credit_spread" in library
    assert "cov_breadth" in library
    assert library["regime_duration_days"].max() > 1


def test_covariate_matching_prefers_blocks_that_resemble_latest_state() -> None:
    index = pd.bdate_range("2025-01-02", periods=160)
    returns = pd.Series([0.003] * 70 + [-0.012] * 30 + [0.0025] * 60, index=index)
    covariates = pd.DataFrame(
        {
            "volatility": [0.15] * 70 + [0.85] * 30 + [0.18] * 60,
            "credit": [0.25] * 70 + [0.90] * 30 + [0.28] * 60,
        },
        index=index,
    )
    base_config = ForwardSimulationConfig(
        horizon_years=1,
        trading_days_per_year=40,
        paths=80,
        block_days=5,
        random_seed=11,
        min_regime_observations=1,
        covariate_match_weight=0.0,
    )
    matched_config = ForwardSimulationConfig(
        horizon_years=1,
        trading_days_per_year=40,
        paths=80,
        block_days=5,
        random_seed=11,
        min_regime_observations=1,
        covariate_match_weight=1.0,
        covariate_match_temperature=0.25,
    )

    unweighted = simulate_regime_conditioned_paths(
        returns,
        covariates=covariates,
        config=base_config,
    )
    matched = simulate_regime_conditioned_paths(
        returns,
        covariates=covariates,
        config=matched_config,
    )

    assert (
        matched["mean_covariate_match_distance"].mean()
        < unweighted["mean_covariate_match_distance"].mean()
    )


def test_factor_conditioned_paths_use_factor_model_outputs() -> None:
    index = pd.bdate_range("2025-01-02", periods=180)
    market = pd.Series([0.001, 0.002, -0.001, 0.003, -0.002] * 36, index=index)
    credit = pd.Series([0.0005, 0.001, -0.002, 0.0015, -0.001] * 36, index=index)
    factor_returns = pd.DataFrame({"market": market, "credit": credit}, index=index)
    residual = pd.Series([0.0002, -0.0001, 0.0001, 0.0, 0.0003] * 36, index=index)
    strategy_returns = 0.0001 + 0.7 * market - 0.2 * credit + residual
    config = ForwardSimulationConfig(
        horizon_years=1,
        trading_days_per_year=30,
        paths=30,
        block_days=5,
        random_seed=13,
        min_regime_observations=1,
    )

    paths = simulate_factor_conditioned_paths(
        strategy_returns,
        factor_returns,
        config=config,
    )
    summary = summarize_forward_simulation(paths, config=config)

    assert len(paths) == 30
    assert paths["factor_model_r_squared"].notna().all()
    assert paths["factor_count"].eq(2).all()
    assert summary["factor_model_r_squared"] is not None


def test_simulate_regime_conditioned_paths_uses_monthly_contributions() -> None:
    returns = pd.Series([0.0] * 80)
    config = ForwardSimulationConfig(
        horizon_years=1,
        trading_days_per_year=12,
        paths=5,
        block_days=1,
        starting_account_value=100.0,
        annual_contribution=12.0,
        random_seed=3,
        min_regime_observations=1,
    )

    paths = simulate_regime_conditioned_paths(returns, config=config)
    summary = summarize_forward_simulation(paths, config=config)

    assert paths["terminal_wealth"].tolist() == pytest.approx([112.0] * 5)
    assert summary["terminal_wealth_p50"] == pytest.approx(112.0)


def test_forward_simulation_empty_inputs_return_empty_summary() -> None:
    paths = simulate_regime_conditioned_paths(pd.Series(dtype=float))
    summary = summarize_forward_simulation(paths)

    assert paths.empty
    assert summary["paths"] == 0
    assert summary["terminal_wealth_p50"] is None


def test_rolling_origin_simulation_backtest_scores_calibration() -> None:
    index = pd.bdate_range("2024-01-02", periods=220)
    returns = pd.Series(
        [0.002] * 70 + [-0.010] * 20 + [0.003] * 70 + [0.0005] * 60,
        index=index,
    )
    scenario_history = pd.DataFrame(
        [
            {
                "origin_date": index[80],
                "risk_bucket": "risk_on",
                "probability": 0.70,
                "horizon": "1m",
            },
            {
                "origin_date": index[80],
                "risk_bucket": "transition",
                "probability": 0.30,
                "horizon": "1m",
            },
        ]
    )
    config = ForwardSimulationValidationConfig(
        origin_frequency="monthly",
        horizons=(("1m", 20), ("3m", 60)),
        min_train_days=60,
        paths=30,
        block_days=5,
        random_seed=17,
        min_regime_observations=1,
    )

    validation = rolling_origin_simulation_backtest(
        returns,
        scenario_history=scenario_history,
        config=config,
    )
    summary = summarize_simulation_validation(validation)

    assert not validation.empty
    assert {"1m", "3m"}.issuperset(set(validation["horizon"]))
    assert validation["train_days"].min() >= 60
    assert validation["simulated_p10_return"].notna().all()
    assert validation["simulated_p50_return"].notna().all()
    assert validation["simulated_p90_return"].notna().all()
    assert validation["realized_in_interval"].isin([True, False]).all()
    assert set(validation["simulated_launch_decision"]).issubset({"wait", "ramp_in", "full_launch"})
    assert validation["launch_action_error"].between(0, 2).all()
    assert validation["launch_overrisk"].isin([True, False]).all()
    assert summary["rows"] == len(validation)
    assert summary["target_coverage"] == pytest.approx(0.60)
    assert summary["launch_action_score"] is not None
    assert 0.0 <= summary["launch_action_score"] <= 1.0
    assert summary["launch_overrisk_rate"] is not None
    assert summary["constructive_capture_rate"] is not None
    assert summary["validity_read"] in {
        "limited_sample",
        "interval_too_narrow",
        "interval_too_wide",
        "too_bullish",
        "too_bearish",
        "drawdown_miscalibrated",
        "return_bands_calibrated__action_checks_not_ready",
        "return_bands_calibrated__action_checks_marginal",
        "return_bands_and_action_checks_ready_for_research",
    }


def test_rolling_origin_simulation_checkpoint_resumes_without_recomputation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    index = pd.bdate_range("2024-01-02", periods=160)
    returns = pd.Series([0.001] * 70 + [-0.006] * 20 + [0.002] * 70, index=index)
    config = ForwardSimulationValidationConfig(
        origin_frequency="quarterly",
        horizons=(("1m", 20),),
        min_train_days=50,
        paths=8,
        block_days=5,
        random_seed=31,
        min_regime_observations=1,
    )
    checkpoint = tmp_path / "simulation_checkpoint.csv"
    original = rolling_origin_simulation_backtest(
        returns,
        config=config,
        checkpoint_path=checkpoint,
        checkpoint_every_origins=1,
    )

    def fail_if_recomputed(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("completed origins should be loaded from the checkpoint")

    monkeypatch.setattr(
        "trade_bot.research.forward_simulation.simulate_regime_conditioned_paths",
        fail_if_recomputed,
    )
    resumed = rolling_origin_simulation_backtest(
        returns,
        config=config,
        checkpoint_path=checkpoint,
        resume=True,
        checkpoint_every_origins=1,
    )

    resumed_comparable = resumed.astype(object).where(resumed.notna(), None)
    original_comparable = original.astype(object).where(original.notna(), None)
    pd.testing.assert_frame_equal(resumed_comparable, original_comparable, check_dtype=False)
    assert checkpoint.exists()
    assert checkpoint.with_suffix(".csv.meta.json").exists()


def test_rolling_origin_simulation_checkpoint_rejects_changed_inputs(tmp_path) -> None:
    index = pd.bdate_range("2024-01-02", periods=140)
    returns = pd.Series([0.001] * 140, index=index)
    config = ForwardSimulationValidationConfig(
        origin_frequency="quarterly",
        horizons=(("1m", 20),),
        min_train_days=50,
        paths=5,
        min_regime_observations=1,
    )
    checkpoint = tmp_path / "simulation_checkpoint.csv"
    rolling_origin_simulation_backtest(
        returns,
        config=config,
        checkpoint_path=checkpoint,
        checkpoint_every_origins=1,
    )

    changed = returns.copy()
    changed.iloc[-1] = -0.01
    with pytest.raises(ValueError, match="do not match"):
        rolling_origin_simulation_backtest(
            changed,
            config=config,
            checkpoint_path=checkpoint,
            resume=True,
        )


def test_simulation_validity_does_not_hide_weak_action_accuracy() -> None:
    validation = pd.DataFrame(
        [
            {
                "origin_date": f"2024-{month:02d}-01",
                "horizon": "3m",
                "realized_in_interval": index % 2 == 0,
                "target_interval_coverage": 0.5,
                "p50_error": 0.01,
                "simulated_severe_drawdown_probability": 0.2,
                "realized_severe_drawdown": False,
                "simulated_launch_decision": "full_launch",
                "realized_launch_decision": "wait" if index < 8 else "full_launch",
                "launch_action_error": 2 if index < 8 else 0,
                "launch_overrisk": index < 8,
                "launch_underrisk": False,
                "avoided_bad_launch_action": False,
                "captured_constructive_launch": True,
            }
            for index, month in enumerate(range(1, 13))
        ]
    )

    summary = summarize_simulation_validation(validation)

    assert summary["distribution_calibration_read"] == "return_bands_calibrated_for_research"
    assert summary["action_readiness_read"] == "action_checks_not_ready"
    assert summary["validity_read"] == "return_bands_calibrated__action_checks_not_ready"


def test_rolling_origin_simulation_backtest_accepts_utc_scenario_history_dates() -> None:
    index = pd.bdate_range("2024-01-02", periods=160)
    returns = pd.Series([0.0015] * 70 + [-0.006] * 20 + [0.002] * 70, index=index)
    scenario_history = pd.DataFrame(
        [
            {
                "snapshot_time": pd.Timestamp(index[70], tz="UTC"),
                "market_date": index[70].date(),
                "risk_bucket": "risk_on",
                "probability": 0.65,
                "horizon": "1m",
            },
            {
                "snapshot_time": pd.Timestamp(index[70], tz="UTC"),
                "market_date": index[70].date(),
                "risk_bucket": "transition",
                "probability": 0.35,
                "horizon": "1m",
            },
        ]
    )
    config = ForwardSimulationValidationConfig(
        origin_frequency="monthly",
        horizons=(("1m", 20),),
        min_train_days=50,
        paths=20,
        block_days=5,
        random_seed=29,
        min_regime_observations=1,
    )

    validation = rolling_origin_simulation_backtest(
        returns,
        scenario_history=scenario_history,
        config=config,
    )

    assert not validation.empty
    assert validation["simulated_p50_return"].notna().all()


def test_simulation_validation_by_horizon_summarizes_decision_quality() -> None:
    validation = pd.DataFrame(
        [
            {
                "origin_date": "2024-01-31",
                "horizon": "1m",
                "horizon_days": 20,
                "realized_in_interval": True,
                "target_interval_coverage": 0.5,
                "p50_error": 0.01,
                "simulated_severe_drawdown_probability": 0.10,
                "realized_severe_drawdown": False,
                "simulated_launch_decision": "ramp_in",
                "realized_launch_decision": "full_launch",
                "launch_action_error": 1,
                "launch_overrisk": False,
                "launch_underrisk": True,
                "avoided_bad_launch_action": True,
                "captured_constructive_launch": True,
            },
            {
                "origin_date": "2024-02-29",
                "horizon": "3m",
                "horizon_days": 60,
                "realized_in_interval": False,
                "target_interval_coverage": 0.5,
                "p50_error": -0.05,
                "simulated_severe_drawdown_probability": 0.30,
                "realized_severe_drawdown": True,
                "simulated_launch_decision": "full_launch",
                "realized_launch_decision": "wait",
                "launch_action_error": 2,
                "launch_overrisk": True,
                "launch_underrisk": False,
                "avoided_bad_launch_action": False,
                "captured_constructive_launch": None,
            },
        ]
    )

    summary = summarize_simulation_validation_by_horizon(validation).set_index("horizon")

    assert set(summary.index) == {"1m", "3m"}
    assert summary.loc["1m", "launch_action_score"] == pytest.approx(0.5)
    assert summary.loc["3m", "launch_overrisk_rate"] == pytest.approx(1.0)


def test_rolling_origin_simulation_backtest_ignores_undated_scenario_history() -> None:
    index = pd.bdate_range("2024-01-02", periods=160)
    returns = pd.Series([0.001] * 60 + [-0.005] * 20 + [0.002] * 80, index=index)
    undated_scenario_history = pd.DataFrame(
        [
            {"risk_bucket": "risk_off", "probability": 1.0, "horizon": "1m"},
            {"risk_bucket": "risk_on", "probability": 0.0, "horizon": "1m"},
        ]
    )
    config = ForwardSimulationValidationConfig(
        origin_frequency="monthly",
        horizons=(("1m", 20),),
        min_train_days=50,
        paths=20,
        block_days=5,
        random_seed=19,
        min_regime_observations=1,
    )

    without_history = rolling_origin_simulation_backtest(returns, config=config)
    with_undated_history = rolling_origin_simulation_backtest(
        returns,
        scenario_history=undated_scenario_history,
        config=config,
    )

    pd.testing.assert_frame_equal(with_undated_history, without_history)


def test_rolling_origin_strategy_rank_validation_scores_predicted_rankings() -> None:
    index = pd.bdate_range("2024-01-02", periods=180)
    strategy_returns = {
        "steady_winner": pd.Series([0.002] * 180, index=index),
        "steady_lagger": pd.Series([0.0002] * 180, index=index),
    }
    config = ForwardSimulationValidationConfig(
        origin_frequency="monthly",
        horizons=(("1m", 20),),
        min_train_days=50,
        paths=20,
        block_days=5,
        random_seed=23,
        min_regime_observations=1,
    )

    rank_validation = rolling_origin_strategy_rank_validation(
        strategy_returns,
        config=config,
    )
    summary = summarize_strategy_rank_validation(rank_validation)

    assert not rank_validation.empty
    assert set(rank_validation["strategy"]) == {"steady_winner", "steady_lagger"}
    top_rows = rank_validation[rank_validation["simulated_rank"] == 1]
    assert set(top_rows["predicted_top_strategy"]) == {"steady_winner"}
    assert summary["top_strategy_hit_rate"] == pytest.approx(1.0)
    assert summary["ranking_read"] in {
        "limited_sample",
        "ranking_signal_useful",
        "ranking_signal_mixed",
    }
