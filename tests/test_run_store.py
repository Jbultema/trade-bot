from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.current_state import CurrentStateRun
from trade_bot.research.event_risk import EventRiskRun
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.signal_inclusion import SignalInclusionRun
from trade_bot.research.trade_decision import TradeDecisionRun
from trade_bot.storage.run_store import RunStore, build_snapshot_fingerprint


def test_run_store_saves_loads_and_lists_snapshots(tmp_path: Path) -> None:
    config_path, events_path, macro_path, news_path = _config_files(tmp_path)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    manifest = store.save_snapshot(
        _baseline_run(),
        config_path=config_path,
        events_path=events_path,
        macro_path=macro_path,
        news_path=news_path,
    )
    loaded = store.load_latest_snapshot(
        fingerprint=build_snapshot_fingerprint(config_path, events_path, macro_path, news_path),
        require_matching_config=True,
    )
    snapshots = store.list_snapshots()

    assert loaded is not None
    loaded_run, loaded_manifest = loaded
    assert loaded_manifest.run_id == manifest.run_id
    assert loaded_run.current_state.market_date == "2026-06-17"
    assert snapshots.iloc[0]["run_id"] == manifest.run_id
    assert Path(manifest.artifact_path).exists()


def test_run_store_tracks_snapshot_jobs(tmp_path: Path) -> None:
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    job = store.create_job(["python", "-V"], tmp_path / "jobs" / "test.log")
    store.mark_job_running(job.job_id)
    store.mark_job_completed(job.job_id, "run-1")
    jobs = store.list_jobs()

    assert jobs.iloc[0]["job_id"] == job.job_id
    assert jobs.iloc[0]["status"] == "completed"
    assert jobs.iloc[0]["run_id"] == "run-1"


def _config_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    paths = (
        tmp_path / "baseline.yaml",
        tmp_path / "events.yaml",
        tmp_path / "macro.yaml",
        tmp_path / "news.yaml",
    )
    for path in paths:
        path.write_text("test: true\n", encoding="utf-8")
    return paths


def _baseline_run() -> BaselineRun:
    index = pd.bdate_range("2026-06-01", periods=4)
    prices = pd.DataFrame({"SPY": [100.0, 101.0, 102.0, 103.0]}, index=index)
    weights = pd.DataFrame({"SPY": [1.0, 1.0, 1.0, 1.0]}, index=index)
    returns = pd.Series([0.0, 0.01, 0.01, 0.01], index=index)
    result = BacktestResult(
        name="demo",
        equity=100.0 * (1.0 + returns).cumprod(),
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )
    current_state = CurrentStateRun(
        market_date="2026-06-17",
        risk_score=0.2,
        risk_status="green",
        risk_summary="Green test state.",
        market_health=pd.DataFrame(),
        momentum_state=pd.DataFrame(),
        confirmation_matrix=pd.DataFrame(),
        strategy_alerts=pd.DataFrame(),
        scenario_outlook=pd.DataFrame(),
        scenario_lattice=pd.DataFrame(),
        scenario_drivers=pd.DataFrame(),
        macro_signals=pd.DataFrame(),
        macro_category_summary=pd.DataFrame(),
        signal_coverage=pd.DataFrame(),
        data_quality=pd.DataFrame(),
    )
    trade_decision = TradeDecisionRun(
        summary=pd.DataFrame(
            [
                {
                    "recommended_action": "HOLD",
                    "risk_budget_multiplier": 1.0,
                }
            ]
        ),
        position_plan=pd.DataFrame(),
        evidence=pd.DataFrame(),
        scenario_links=pd.DataFrame(),
    )
    return BaselineRun(
        prices=prices,
        macro_data=pd.DataFrame(),
        macro_catalog=(),
        results={"demo": result},
        metrics=pd.DataFrame(),
        rolling_windows=pd.DataFrame(),
        window_summary=pd.DataFrame(),
        calendar_metrics=pd.DataFrame(),
        calendar_returns=pd.DataFrame(),
        current_state=current_state,
        event_risk=EventRiskRun(
            events=(),
            asset_event_returns=pd.DataFrame(),
            strategy_event_returns=pd.DataFrame(),
            event_summary=pd.DataFrame(),
            scenario_playbook=pd.DataFrame(),
            current_event_scenarios=pd.DataFrame(),
        ),
        news_monitor=NewsMonitorRun(
            items=(),
            triage=pd.DataFrame(),
            source_health=pd.DataFrame(),
            activated_events=(),
            activation_threshold=0.8,
            lookback_days=7,
        ),
        signal_inclusion=SignalInclusionRun(
            summary=pd.DataFrame(),
            pressure=pd.DataFrame(),
            results={},
            metrics=pd.DataFrame(),
            window_summary=pd.DataFrame(),
        ),
        trade_decision=trade_decision,
    )
