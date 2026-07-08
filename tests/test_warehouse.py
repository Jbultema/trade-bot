from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import trade_bot.storage.warehouse as warehouse_module
from trade_bot.config import ExecutionConfig, StrategyConfig
from trade_bot.storage.warehouse import TradingWarehouse


def test_warehouse_migrates_experiments_seeds_windows_and_values_snapshot(tmp_path) -> None:
    experiment_dir = tmp_path / "experiments"
    iteration_dir = experiment_dir / "iteration_01"
    iteration_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy": "candidate_alpha",
                "phase": "operating_system",
                "family": "ai_beta",
                "role": "satellite",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.80,
                "robustness_score": 0.70,
                "cagr": 0.12,
                "sharpe": 1.10,
                "max_drawdown": -0.12,
                "calmar": 1.00,
                "walk_forward_median_cagr": 0.10,
                "walk_forward_positive_rate": 0.90,
                "left_tail_regime_return": -0.08,
                "hypothesis": "Test promoted strategy.",
            },
            {
                "strategy": "fragile_alpha",
                "phase": "operating_system",
                "family": "fragile",
                "role": "candidate",
                "promotion_decision": "reject_left_tail",
                "promotion_score": 0.70,
                "robustness_score": 0.20,
                "cagr": 0.20,
                "sharpe": 1.20,
                "max_drawdown": -0.45,
                "calmar": 0.44,
                "walk_forward_median_cagr": -0.03,
                "walk_forward_positive_rate": 0.40,
                "left_tail_regime_return": -0.40,
                "hypothesis": "Rejected by left tail.",
            },
        ]
    ).to_csv(iteration_dir / "scorecard.csv", index=False)

    warehouse = TradingWarehouse(tmp_path / "trade_bot.duckdb")
    migrated = warehouse.migrate_experiment_outputs(experiment_dir)

    assert {result.table_name for result in migrated} == {"experiment_scorecard"}
    scorecard = warehouse.read_table("experiment_scorecard")
    labels = scorecard.set_index("strategy")["overfit_risk_label"].to_dict()
    statuses = scorecard.set_index("strategy")["research_status"].to_dict()
    assert labels["candidate_alpha"] == "low"
    assert labels["fragile_alpha"] in {"high", "critical"}
    assert statuses["candidate_alpha"] == "operational_candidate"
    assert statuses["fragile_alpha"] == "pruned_dead_end"
    assert "selection_adjusted_promotion_score" in scorecard

    seeded = warehouse.seed_monitoring_windows_from_registry(
        account="shadow",
        top_n=1,
        start_date="2026-06-16",
    )

    assert len(seeded) == 1
    assert seeded[0].strategy_name == "candidate_alpha"

    baseline_run = SimpleNamespace(
        current_state=SimpleNamespace(market_date="2026-06-18"),
        prices=pd.DataFrame(
            {"SPY": [100.0, 101.0, 102.0]},
            index=pd.to_datetime(["2026-06-16", "2026-06-17", "2026-06-18"]),
        ),
        results={
            "candidate_alpha": SimpleNamespace(
                equity=pd.Series(
                    [100.0, 101.0, 103.0],
                    index=pd.to_datetime(["2026-06-16", "2026-06-17", "2026-06-18"]),
                ),
                returns=pd.Series(
                    [0.0, 0.01, 0.01980198],
                    index=pd.to_datetime(["2026-06-16", "2026-06-17", "2026-06-18"]),
                ),
                weights=pd.DataFrame(
                    {"QQQ": [0.5, 0.5, 0.5], "BIL": [0.5, 0.5, 0.5]},
                    index=pd.to_datetime(["2026-06-16", "2026-06-17", "2026-06-18"]),
                ),
            )
        },
    )

    valued_rows = warehouse.save_daily_valuations_from_snapshot(baseline_run)
    first_champion = warehouse.champion_challenger_frame()
    second_rows = warehouse.save_daily_valuations_from_snapshot(
        baseline_run,
        market_date="2026-06-19",
    )
    champion = warehouse.champion_challenger_frame()

    assert valued_rows == 1
    assert second_rows == 1
    assert first_champion.iloc[0]["equity"] == pytest.approx(10_300.0)
    assert first_champion.iloc[0]["cumulative_return"] == pytest.approx(0.03)
    assert first_champion.iloc[0]["forward_status"] == "ahead_of_benchmark"
    assert champion.iloc[0]["strategy_name"] == "candidate_alpha"
    assert champion.iloc[0]["forward_status"] == "ahead_of_benchmark"
    assert champion.iloc[0]["validation_tier"] == "paper_champion_candidate"
    assert champion.iloc[0]["stocks_percent_of_max_sleeve"] == pytest.approx(0.5 / 0.6)
    assert champion.iloc[0]["defensive_percent_of_max_sleeve"] == pytest.approx(0.5)
    assert json.loads(str(champion.iloc[0]["latest_weights_json"])) == {"BIL": 0.5, "QQQ": 0.5}


def test_warehouse_persists_simulation_validation_history(tmp_path) -> None:
    warehouse = TradingWarehouse(tmp_path / "trade_bot.duckdb")
    validation = pd.DataFrame(
        [
            {
                "origin_date": "2025-01-31",
                "horizon": "1m",
                "horizon_days": 20,
                "train_days": 252,
                "paths": 100,
                "realized_return": 0.02,
                "realized_max_drawdown": -0.03,
                "realized_severe_drawdown": False,
                "simulated_p10_return": -0.01,
                "simulated_p50_return": 0.015,
                "simulated_p90_return": 0.04,
                "target_interval_coverage": 0.80,
                "realized_in_interval": True,
                "p50_error": -0.005,
                "p50_abs_error": 0.005,
                "simulated_severe_drawdown_probability": 0.10,
                "severe_drawdown_probability_error": 0.10,
                "simulated_launch_decision": "launch",
                "realized_launch_decision": "launch",
            }
        ]
    )
    ablation = pd.DataFrame(
        [
            {
                "variant": "duration_covariate",
                "label": "Duration + covariate matching",
                "uses_duration_aware_transitions": True,
                "uses_covariate_matching": True,
                "uses_factor_proxy": False,
                "rows": 1,
                "origins": 1,
                "horizons": 1,
                "interval_coverage": 1.0,
                "target_coverage": 0.8,
                "coverage_error": 0.2,
                "median_error_mean": -0.005,
                "median_abs_error": 0.005,
                "severe_drawdown_brier": 0.01,
                "launch_decision_accuracy": 1.0,
                "validity_read": "calibrated_enough_for_research",
            }
        ]
    )

    validation_run_id = warehouse.save_simulation_validation_run(
        snapshot_run_id="snapshot-1",
        market_date="2025-01-31",
        strategy="strategy_a",
        reference_strategies="strategy_b",
        horizons="1m=20",
        origin_frequency="monthly",
        min_train_days=252,
        paths=100,
        block_days=21,
        scenario_history_path="",
        validation_output_path="validation.csv",
        ablation_output_path="ablation.csv",
        rank_output_path="rank.csv",
        validation_summary={
            "rows": 1,
            "origins": 1,
            "horizons": 1,
            "interval_coverage": 1.0,
            "target_coverage": 0.8,
            "coverage_error": 0.2,
            "median_abs_error": 0.005,
            "launch_decision_accuracy": 1.0,
            "validity_read": "calibrated_enough_for_research",
        },
        validation=validation,
        ablation_summary=ablation,
    )

    runs = warehouse.simulation_validation_runs()
    metrics = warehouse.simulation_validation_metrics(validation_run_id=validation_run_id)
    ablation_metrics = warehouse.simulation_validation_metrics(
        validation_run_id=validation_run_id,
        metric_scope="ablation_summary",
    )

    assert runs.iloc[0]["validation_run_id"] == validation_run_id
    assert runs.iloc[0]["strategy"] == "strategy_a"
    assert set(metrics["metric_scope"]) == {
        "primary_summary",
        "rolling_origin",
        "ablation_summary",
    }
    assert ablation_metrics.iloc[0]["variant"] == "duration_covariate"
    assert bool(ablation_metrics.iloc[0]["uses_covariate_matching"])


def test_warehouse_surfaces_and_seeds_top_5_experiment_candidates(tmp_path) -> None:
    experiment_dir = tmp_path / "experiments"
    iteration_dir = experiment_dir / "iteration_40"
    iteration_dir.mkdir(parents=True)
    rows = []
    for index in range(30):
        rows.append(
            {
                "strategy": f"candidate_{index:02d}",
                "phase": "operating_system",
                "family": "rotation",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 1.0 - index / 100.0,
                "robustness_score": 0.8 - index / 200.0,
                "cagr": 0.12 - index / 1000.0,
                "sharpe": 1.0,
                "max_drawdown": -0.12,
                "calmar": 1.0 - index / 100.0,
                "average_turnover": 0.04,
                "walk_forward_median_cagr": 0.08,
                "walk_forward_positive_rate": 0.8,
                "left_tail_regime_return": -0.08,
                "hypothesis": "Ranked experiment candidate.",
            }
        )
    pd.DataFrame(rows).to_csv(iteration_dir / "scorecard.csv", index=False)

    warehouse = TradingWarehouse(tmp_path / "trade_bot.duckdb")
    warehouse.migrate_experiment_outputs(experiment_dir)

    top_candidates = warehouse.top_monitoring_candidates()

    assert len(top_candidates) == 5
    assert top_candidates["rank"].tolist() == list(range(1, 6))
    assert top_candidates.iloc[0]["strategy_name"] == "candidate_00"
    assert top_candidates.iloc[-1]["strategy_name"] == "candidate_04"
    assert set(top_candidates["source"]) == {"experiment_scorecard"}
    assert int(top_candidates["is_active_window"].sum()) == 0
    assert set(top_candidates["monitoring_state"]) == {"available_research_only"}

    seeded = warehouse.seed_monitoring_windows_from_registry(
        account="shadow",
        start_date="2026-06-18",
    )
    windows = warehouse.list_monitoring_windows(status="active")
    active_top_candidates = warehouse.top_monitoring_candidates()

    assert len(seeded) == 5
    assert len(windows) == 5
    assert int((windows["window_role"] == "champion").sum()) == 1
    assert int((windows["window_role"] == "challenger").sum()) == 4
    assert int(active_top_candidates["is_active_window"].sum()) == 5
    assert set(active_top_candidates["monitoring_state"]) == {"active_research_only"}

    additional_seeded = warehouse.seed_monitoring_windows_from_registry(
        account="shadow",
        top_n=6,
        start_date="2026-06-18",
    )
    expanded_windows = warehouse.list_monitoring_windows(status="active")

    assert len(additional_seeded) == 1
    assert additional_seeded[0].role == "challenger"
    assert int((expanded_windows["window_role"] == "champion").sum()) == 1
    assert int((expanded_windows["window_role"] == "challenger").sum()) == 5


def test_warehouse_keeps_only_core_reference_portfolios_visible_for_default_monitoring(
    tmp_path,
) -> None:
    experiment_dir = tmp_path / "experiments"
    iteration_dir = experiment_dir / "iteration_41"
    iteration_dir.mkdir(parents=True)
    rows = [
        {
            "strategy": "candidate_core",
            "phase": "operating_system",
            "family": "rotation",
            "role": "candidate",
            "promotion_decision": "promote_candidate",
            "promotion_score": 0.95,
            "robustness_score": 0.80,
            "cagr": 0.12,
            "sharpe": 1.0,
            "max_drawdown": -0.12,
            "calmar": 1.0,
            "average_turnover": 0.04,
            "walk_forward_positive_rate": 0.8,
            "left_tail_regime_return": -0.08,
            "hypothesis": "Top tactical candidate.",
        },
        {
            "strategy": "i41_ref_us_60_40",
            "phase": "reference",
            "family": "reference_portfolio",
            "role": "reference_portfolio",
            "promotion_decision": "reject_or_hold_for_reference",
            "promotion_score": 0.40,
            "robustness_score": 0.50,
            "cagr": 0.07,
            "sharpe": 0.7,
            "max_drawdown": -0.25,
            "calmar": 0.28,
            "average_turnover": 0.01,
            "walk_forward_positive_rate": 0.6,
            "left_tail_regime_return": -0.12,
            "hypothesis": "Reference policy.",
        },
        {
            "strategy": "i41_ref_all_weather",
            "phase": "reference",
            "family": "reference_portfolio",
            "role": "reference_portfolio",
            "promotion_decision": "reject_or_hold_for_reference",
            "promotion_score": 0.45,
            "robustness_score": 0.55,
            "cagr": 0.06,
            "sharpe": 0.8,
            "max_drawdown": -0.18,
            "calmar": 0.33,
            "average_turnover": 0.01,
            "walk_forward_positive_rate": 0.7,
            "left_tail_regime_return": -0.09,
            "hypothesis": "Reference policy.",
        },
    ]
    pd.DataFrame(rows).to_csv(iteration_dir / "scorecard.csv", index=False)

    warehouse = TradingWarehouse(tmp_path / "trade_bot.duckdb")
    warehouse.migrate_experiment_outputs(experiment_dir)

    top_candidates = warehouse.top_monitoring_candidates()
    reference_candidates = warehouse.reference_monitoring_candidates()
    seeded = warehouse.seed_monitoring_windows_from_registry(
        account="shadow",
        start_date="2026-06-18",
    )
    windows = warehouse.list_monitoring_windows(status="active")

    assert top_candidates.iloc[0]["strategy_name"] == "candidate_core"
    assert set(reference_candidates["strategy_name"]) == {"i41_ref_us_60_40"}
    assert "i41_ref_all_weather" not in set(reference_candidates["strategy_name"])
    assert len(seeded) == 2
    assert int((windows["window_role"] == "champion").sum()) == 1
    assert int((windows["window_role"] == "reference").sum()) == 1


def test_warehouse_manually_monitors_and_values_experiment_candidate(tmp_path) -> None:
    experiment_dir = tmp_path / "experiments"
    iteration_dir = experiment_dir / "iteration_42"
    iteration_dir.mkdir(parents=True)
    strategy = StrategyConfig(type="buy_hold", tickers=["QQQ"])
    pd.DataFrame(
        [
            {
                "strategy": "manual_candidate",
                "phase": "operating_system",
                "family": "runtime_test",
                "role": "candidate",
                "parent": "",
                "hypothesis": "Can be monitored manually.",
                "scenario_sizing": "",
                "scenario_sizing_json": "",
                "strategy_json": json.dumps(strategy.model_dump(mode="json")),
            }
        ]
    ).to_csv(iteration_dir / "candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy": "manual_candidate",
                "phase": "operating_system",
                "family": "runtime_test",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.90,
                "robustness_score": 0.80,
                "cagr": 0.12,
                "sharpe": 1.0,
                "max_drawdown": -0.12,
                "calmar": 1.0,
                "average_turnover": 0.02,
                "walk_forward_positive_rate": 0.8,
                "left_tail_regime_return": -0.08,
                "hypothesis": "Can be monitored manually.",
            }
        ]
    ).to_csv(iteration_dir / "scorecard.csv", index=False)

    warehouse = TradingWarehouse(tmp_path / "trade_bot.duckdb")
    warehouse.migrate_experiment_outputs(experiment_dir)
    seedable = warehouse.top_monitoring_candidates(limit=1)

    assert bool(seedable.iloc[0]["is_active_window"]) is False
    assert seedable.iloc[0]["monitoring_state"] == "available_to_seed_and_value"

    monitored = warehouse.monitor_strategy(
        "manual_candidate",
        role="challenger",
        account="paper_core",
        capital_base=10_000.0,
        start_date="2026-06-15",
    )
    separate_champion = warehouse.monitor_strategy(
        "manual_candidate",
        role="champion",
        account="paper_satellite",
        capital_base=5_000.0,
        start_date="2026-06-15",
    )
    updated = warehouse.update_monitoring_window(
        monitored.window_id,
        role="champion",
        capital_base=10_000.0,
        demote_other_champions=True,
    )
    windows = warehouse.list_monitoring_windows(status="active")

    assert updated is True
    assert len(windows) == 2
    assert set(windows["window_role"]) == {"champion"}
    assert set(windows["capital_base"].astype(float)) == {10_000.0, 5_000.0}
    assert monitored.window_id != separate_champion.window_id

    baseline_run = SimpleNamespace(
        current_state=SimpleNamespace(market_date="2026-06-18"),
        prices=pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0, 103.0],
                "QQQ": [100.0, 102.0, 104.0, 108.0],
            },
            index=pd.to_datetime(["2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18"]),
        ),
        results={},
    )

    first_rows = warehouse.save_daily_valuations_from_snapshot(
        baseline_run,
        market_date="2026-06-18",
        execution=ExecutionConfig(initial_capital=100_000.0, rebalance="D", signal_lag_days=1),
    )
    second_rows = warehouse.save_daily_valuations_from_snapshot(
        baseline_run,
        market_date="2026-06-19",
        execution=ExecutionConfig(initial_capital=100_000.0, rebalance="D", signal_lag_days=1),
    )
    valued = warehouse.champion_challenger_frame()

    assert first_rows == 2
    assert second_rows == 2
    assert set(valued["strategy_name"]) == {"manual_candidate"}
    assert valued["valuation_date"].notna().all()
    assert valued["equity"].min() > 5_000.0


def test_warehouse_resets_monitoring_start_dates_and_reanchors_valuations(tmp_path) -> None:
    experiment_dir = tmp_path / "experiments"
    iteration_dir = experiment_dir / "iteration_43"
    iteration_dir.mkdir(parents=True)
    strategy = StrategyConfig(type="buy_hold", tickers=["QQQ"])
    pd.DataFrame(
        [
            {
                "strategy": "reset_candidate",
                "phase": "operating_system",
                "family": "runtime_test",
                "role": "candidate",
                "parent": "",
                "hypothesis": "Can reset monitoring start.",
                "scenario_sizing": "",
                "scenario_sizing_json": "",
                "strategy_json": json.dumps(strategy.model_dump(mode="json")),
            }
        ]
    ).to_csv(iteration_dir / "candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy": "reset_candidate",
                "phase": "operating_system",
                "family": "runtime_test",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.90,
                "robustness_score": 0.80,
                "cagr": 0.12,
                "sharpe": 1.0,
                "max_drawdown": -0.12,
                "calmar": 1.0,
                "average_turnover": 0.02,
                "walk_forward_positive_rate": 0.8,
                "left_tail_regime_return": -0.08,
                "hypothesis": "Can reset monitoring start.",
            }
        ]
    ).to_csv(iteration_dir / "scorecard.csv", index=False)

    warehouse = TradingWarehouse(tmp_path / "trade_bot.duckdb")
    warehouse.migrate_experiment_outputs(experiment_dir)
    warehouse.monitor_strategy(
        "reset_candidate",
        role="challenger",
        account="paper_core",
        capital_base=10_000.0,
        start_date="2026-06-18",
    )
    baseline_run = SimpleNamespace(
        current_state=SimpleNamespace(market_date="2026-06-18"),
        prices=pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0, 103.0],
                "QQQ": [100.0, 102.0, 104.0, 108.0],
            },
            index=pd.to_datetime(["2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18"]),
        ),
        results={
            "reset_candidate": SimpleNamespace(
                equity=pd.Series(
                    [100.0, 102.0, 104.0, 108.0],
                    index=pd.to_datetime(["2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18"]),
                ),
                returns=pd.Series(
                    [0.0, 0.02, 0.0196078431, 0.0384615385],
                    index=pd.to_datetime(["2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18"]),
                ),
                weights=pd.DataFrame(
                    {"QQQ": [1.0, 1.0, 1.0, 1.0]},
                    index=pd.to_datetime(["2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18"]),
                ),
            )
        },
    )

    first_rows = warehouse.save_daily_valuations_from_snapshot(
        baseline_run,
        market_date="2026-06-18",
        execution=ExecutionConfig(initial_capital=100_000.0, rebalance="D", signal_lag_days=1),
    )
    first = warehouse.champion_challenger_frame()
    reset_rows = warehouse.reset_monitoring_start_dates(
        start_date="2026-06-15",
        mode="paper",
        account="paper_core",
    )
    cleared = warehouse.read_table("strategy_daily_valuations")
    second_rows = warehouse.save_daily_valuations_from_snapshot(
        baseline_run,
        market_date="2026-06-18",
        execution=ExecutionConfig(initial_capital=100_000.0, rebalance="D", signal_lag_days=1),
    )
    second = warehouse.champion_challenger_frame()

    assert first_rows == 1
    assert first.iloc[0]["equity"] == pytest.approx(10_000.0)
    assert reset_rows == 1
    assert cleared.empty
    assert second_rows == 1
    assert second.iloc[0]["start_date"] == "2026-06-15"
    assert second.iloc[0]["equity"] == pytest.approx(10_800.0)


def test_warehouse_values_monitored_strategy_from_artifact_manifest_when_db_manifest_missing(
    tmp_path,
    monkeypatch,
) -> None:
    experiment_dir = tmp_path / "experiments"
    iteration_dir = experiment_dir / "iteration_67"
    iteration_dir.mkdir(parents=True)
    strategy = StrategyConfig(type="buy_hold", tickers=["QQQ"])
    pd.DataFrame(
        [
            {
                "strategy": "artifact_only_candidate",
                "phase": "operating_system",
                "family": "runtime_test",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.90,
                "robustness_score": 0.80,
                "cagr": 0.12,
                "sharpe": 1.0,
                "max_drawdown": -0.12,
                "calmar": 1.0,
                "average_turnover": 0.02,
                "walk_forward_positive_rate": 0.8,
                "left_tail_regime_return": -0.08,
                "hypothesis": "Manifest exists on disk but was not migrated.",
            }
        ]
    ).to_csv(iteration_dir / "scorecard.csv", index=False)

    warehouse = TradingWarehouse(tmp_path / "trade_bot.duckdb")
    warehouse.migrate_experiment_outputs(experiment_dir)
    warehouse.monitor_strategy(
        "artifact_only_candidate",
        role="champion",
        account="paper_core",
        capital_base=10_000.0,
        start_date="2026-06-18",
    )
    pd.DataFrame(
        [
            {
                "strategy": "artifact_only_candidate",
                "phase": "operating_system",
                "family": "runtime_test",
                "role": "candidate",
                "parent": "",
                "hypothesis": "Manifest exists on disk but was not migrated.",
                "scenario_sizing": "",
                "scenario_sizing_json": "",
                "strategy_json": json.dumps(strategy.model_dump(mode="json")),
            }
        ]
    ).to_csv(iteration_dir / "candidates.csv", index=False)
    monkeypatch.setattr(warehouse_module, "DEFAULT_EXPERIMENTS_DIR", experiment_dir)
    monkeypatch.setattr(warehouse_module, "DEFAULT_RESET_EXPERIMENTS_DIR", tmp_path / "missing")

    baseline_run = SimpleNamespace(
        current_state=SimpleNamespace(market_date="2026-06-26"),
        prices=pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0, 103.0],
                "QQQ": [100.0, 102.0, 104.0, 108.0],
            },
            index=pd.to_datetime(["2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26"]),
        ),
        results={},
    )

    valued_rows = warehouse.save_daily_valuations_from_snapshot(
        baseline_run,
        market_date="2026-06-26",
        execution=ExecutionConfig(initial_capital=100_000.0, rebalance="D", signal_lag_days=1),
    )
    valued = warehouse.champion_challenger_frame()

    assert valued_rows == 1
    assert valued.iloc[0]["strategy_name"] == "artifact_only_candidate"
    assert valued.iloc[0]["valuation_date"] == "2026-06-26"
