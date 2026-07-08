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


def test_run_store_prunes_snapshot_artifacts_after_dry_run(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    config_path, events_path, macro_path, news_path = _config_files(tmp_path)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )
    timestamps = iter(
        [
            "2026-07-01T00:00:00+00:00",
            "2026-07-02T00:00:00+00:00",
            "2026-07-03T00:00:00+00:00",
        ]
    )
    monkeypatch.setattr("trade_bot.storage.run_store.utc_now_iso", lambda: next(timestamps))

    manifests = [
        store.save_snapshot(
            _baseline_run(),
            config_path=config_path,
            events_path=events_path,
            macro_path=macro_path,
            news_path=news_path,
        )
        for _ in range(3)
    ]

    dry_run_candidates = store.prune_snapshots(
        keep_latest=1,
        keep_per_market_date=1,
        apply=False,
    )
    assert len(dry_run_candidates) == 2
    assert all(Path(manifest.artifact_path).exists() for manifest in manifests)
    assert len(store.list_snapshots(limit=10)) == 3

    applied_candidates = store.prune_snapshots(
        keep_latest=1,
        keep_per_market_date=1,
        apply=True,
    )
    remaining = store.list_snapshots(limit=10)

    assert len(applied_candidates) == 2
    assert applied_candidates["pruned"].tolist() == [True, True]
    assert remaining["run_id"].tolist() == [manifests[-1].run_id]
    assert Path(manifests[-1].artifact_path).exists()
    assert not Path(manifests[0].artifact_path).exists()
    assert not Path(manifests[1].artifact_path).exists()


def test_run_store_prunes_to_recent_daily_and_older_weekly_snapshots(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    config_path, events_path, macro_path, news_path = _config_files(tmp_path)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )
    market_dates = pd.bdate_range("2026-05-01", "2026-07-15")
    timestamps = iter(
        f"{market_date.date()}T20:00:00+00:00" for market_date in market_dates
    )
    monkeypatch.setattr("trade_bot.storage.run_store.utc_now_iso", lambda: next(timestamps))
    manifests = [
        store.save_snapshot(
            _baseline_run(str(market_date.date())),
            config_path=config_path,
            events_path=events_path,
            macro_path=macro_path,
            news_path=news_path,
        )
        for market_date in market_dates
    ]

    candidates = store.prune_snapshots(
        keep_latest=0,
        keep_per_market_date=1,
        keep_recent_market_days=5,
        keep_weekly_older=1,
        weekly_frequency="W-WED",
        apply=True,
    )
    remaining = store.list_snapshots(limit=100)
    remaining_dates = pd.to_datetime(remaining["market_date"])
    recent_dates = set(market_dates[-5:].date)
    older = remaining_dates[~remaining_dates.dt.date.isin(recent_dates)]

    assert not candidates.empty
    assert set(remaining_dates.dt.date).issuperset(recent_dates)
    assert older.dt.to_period("W-WED").nunique() == len(older)
    assert len(remaining) == len(recent_dates) + market_dates[:-5].to_period("W-WED").nunique()
    assert all(Path(manifest.artifact_path).exists() for manifest in manifests[-5:])


def test_run_store_starts_daily_update_job(tmp_path: Path, monkeypatch: object) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class DummyPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))

    monkeypatch.setattr("trade_bot.storage.run_store.subprocess.Popen", DummyPopen)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    job = store.start_daily_update_job(
        config_path=tmp_path / "baseline.yaml",
        events_path=tmp_path / "events.yaml",
        macro_path=tmp_path / "macro.yaml",
        news_path=tmp_path / "news.yaml",
        report_path=tmp_path / "report.html",
        experiment_dir=tmp_path / "experiments",
        journal_path=tmp_path / "journal.sqlite",
        refresh_data=True,
        refresh_macro=False,
        refresh_news=True,
        migrate_warehouse=True,
        paper_valuation=False,
    )
    jobs = store.list_jobs()

    assert calls
    command = list(calls[0][0][0])
    assert "run-daily-update" in command
    assert "--refresh-data" in command
    assert "--cached-macro" in command
    assert "--refresh-news" in command
    assert "--migrate-warehouse" in command
    assert "--skip-paper-valuation" in command
    assert "--job-id" in command
    assert job.job_id in command
    assert Path(job.log_path).exists()
    assert jobs.iloc[0]["job_id"] == job.job_id
    assert jobs.iloc[0]["status"] == "queued"
    assert "run-daily-update" in str(jobs.iloc[0]["command"])


def test_run_store_starts_targeted_update_jobs(tmp_path: Path, monkeypatch: object) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class DummyPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))

    monkeypatch.setattr("trade_bot.storage.run_store.subprocess.Popen", DummyPopen)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    migration_job = store.start_warehouse_migration_job(
        experiment_dir=tmp_path / "experiments",
        journal_path=tmp_path / "journal.sqlite",
    )
    valuation_job = store.start_paper_valuation_job(config_path=tmp_path / "baseline.yaml")
    seed_job = store.start_monitoring_seed_job(top_n=3, capital_base=5_000.0)
    reset_job = store.start_monitoring_start_reset_job(
        config_path=tmp_path / "baseline.yaml",
        start_date="2026-01-01",
    )
    ml_job = store.start_ml_diagnostics_job(
        config_path=tmp_path / "baseline.yaml",
        output_dir=tmp_path / "ml",
        profile="standard",
        refresh_data=True,
    )

    commands = [list(call[0][0]) for call in calls]
    assert any("migrate-warehouse" in command for command in commands)
    assert any("run-paper-valuation" in command for command in commands)
    assert any("seed-monitoring-windows" in command for command in commands)
    assert any("reset-monitoring-start-date" in command for command in commands)
    assert any("run-ml-diagnostics" in command for command in commands)
    assert any("--refresh-data" in command for command in commands)
    assert all("--job-id" in command for command in commands)
    assert {
        migration_job.job_id,
        valuation_job.job_id,
        seed_job.job_id,
        reset_job.job_id,
        ml_job.job_id,
    } == set(store.list_jobs(limit=10)["job_id"])


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


def _baseline_run(market_date: str = "2026-06-17") -> BaselineRun:
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
        market_date=market_date,
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
