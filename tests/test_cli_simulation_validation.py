from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from trade_bot.cli import app
from trade_bot.storage.warehouse import TradingWarehouse


def test_validate_simulation_engine_command_writes_validation_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    index = pd.bdate_range("2025-01-02", periods=140)
    baseline_run = SimpleNamespace(
        results={
            "strategy_a": SimpleNamespace(
                returns=pd.Series([0.001] * 70 + [-0.004] * 10 + [0.002] * 60, index=index)
            ),
            "strategy_b": SimpleNamespace(returns=pd.Series([0.0005] * 140, index=index)),
        }
    )
    manifest = SimpleNamespace(run_id="run_test", market_date="2025-07-17")
    store_path = tmp_path / "trade_bot.duckdb"

    class FakeRunStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def load_latest_snapshot(self, *args: object, **kwargs: object) -> object:
            return baseline_run, manifest

    monkeypatch.setattr("trade_bot.cli.RunStore", FakeRunStore)

    result = CliRunner().invoke(
        app,
        [
            "validate-simulation-engine",
            "--store",
            str(store_path),
            "--output-dir",
            str(tmp_path),
            "--strategy",
            "strategy_a",
            "--reference-strategies",
            "strategy_b",
            "--horizons",
            "1m=20",
            "--origin-frequency",
            "monthly",
            "--min-train-days",
            "40",
            "--paths",
            "8",
            "--block-days",
            "5",
            "--ablation",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "strategy_a_simulation_validation.csv").exists()
    assert (tmp_path / "strategy_a_simulation_ablation.csv").exists()
    assert (tmp_path / "strategy_rank_validation.csv").exists()
    assert "Simulation Validation: strategy_a" in result.output
    assert "Simulation Model Ablation" in result.output
    assert "Strategy Rank Validation" in result.output
    assert "Saved simulation validation history to DuckDB" in result.output

    warehouse = TradingWarehouse(store_path)
    runs = warehouse.read_table("simulation_validation_runs")
    metrics = warehouse.read_table("simulation_validation_metrics")
    assert len(runs) == 1
    assert runs.iloc[0]["strategy"] == "strategy_a"
    assert runs.iloc[0]["primary_distribution_calibration_read"]
    assert runs.iloc[0]["primary_action_readiness_read"]
    assert {"primary_summary", "horizon_summary", "rolling_origin", "ablation_summary"}.issubset(
        set(metrics["metric_scope"])
    )
    assert "launch_action_score" in metrics
    assert "distribution_calibration_read" in metrics
    assert "action_readiness_read" in metrics
    assert "duration_covariate" in set(metrics["variant"])
