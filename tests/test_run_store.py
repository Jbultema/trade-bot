from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.current_state import CurrentStateRun
from trade_bot.research.event_risk import EventRiskRun
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.signal_inclusion import SignalInclusionRun
from trade_bot.research.trade_decision import TradeDecisionRun
from trade_bot.storage.run_store import RunStore, build_snapshot_fingerprint


def test_read_only_run_store_skips_schema_setup_and_reads_existing_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "trade_bot.duckdb"
    artifact_dir = tmp_path / "snapshots"
    job_log_dir = tmp_path / "jobs"
    RunStore(db_path, artifact_dir=artifact_dir, job_log_dir=job_log_dir)

    def unexpected_schema_setup(_store: object) -> None:
        raise AssertionError("read-only run store must not run schema setup")

    monkeypatch.setattr(RunStore, "_ensure_schema", unexpected_schema_setup)
    reader = RunStore(
        db_path,
        artifact_dir=artifact_dir,
        job_log_dir=job_log_dir,
        read_only=True,
    )

    assert reader.list_snapshots().empty


def test_read_only_run_store_reads_legacy_schema_without_stored_at_utc(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(
            """
            CREATE TABLE run_snapshots (
                run_id VARCHAR,
                created_at_utc VARCHAR,
                status VARCHAR,
                market_date VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO run_snapshots VALUES
            ('older', '2026-07-01T20:00:00+00:00', 'completed', '2026-07-01'),
            ('newer', '2026-07-02T20:00:00+00:00', 'completed', '2026-07-02')
            """
        )
    finally:
        connection.close()

    reader = RunStore(
        db_path,
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
        read_only=True,
    )
    snapshots = reader.list_snapshots(limit=10)

    assert snapshots["run_id"].tolist() == ["newer", "older"]
    assert snapshots["stored_at_utc"].tolist() == snapshots["created_at_utc"].tolist()


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
    provenance = json.loads(loaded_manifest.provenance_json)
    assert provenance["source_tree_sha256"]
    assert provenance["poetry_lock_sha256"]
    assert provenance["price_input"]["frame_sha256"]


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


def test_run_store_tracks_stage_progress_and_snapshot_before_completion(tmp_path: Path) -> None:
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    job = store.create_job(["python", "worker.py"], tmp_path / "jobs" / "test.log")
    store.mark_job_running(job.job_id)
    store.mark_job_stage(job.job_id, "snapshot", "running")
    store.attach_job_run_id(job.job_id, "run-durable")
    store.mark_job_stage(job.job_id, "snapshot", "completed", "run-durable")
    store.mark_job_stage(job.job_id, "paper_valuation", "failed", "valuation unavailable")

    stored_job = store.get_job(job.job_id)
    stages = store.list_job_stages(job.job_id).set_index("stage")

    assert stored_job is not None
    assert stored_job.run_id == "run-durable"
    assert stages.loc["snapshot", "status"] == "completed"
    assert stages.loc["paper_valuation", "status"] == "failed"
    assert stages.loc["paper_valuation", "detail"] == "valuation unavailable"


def test_run_store_migrates_legacy_snapshot_jobs_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "trade_bot.duckdb"
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(
            """
            CREATE TABLE snapshot_jobs (
                job_id VARCHAR PRIMARY KEY,
                created_at_utc VARCHAR NOT NULL,
                started_at_utc VARCHAR NOT NULL,
                completed_at_utc VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                command VARCHAR NOT NULL,
                log_path VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                error_message VARCHAR NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO snapshot_jobs
            VALUES (
                'legacy-job', '2026-01-01T00:00:00+00:00', '', '', 'queued',
                'python -V', 'legacy.log', '', ''
            )
            """
        )
    finally:
        connection.close()

    store = RunStore(
        db_path,
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    migrated = store.get_job("legacy-job")

    assert "process_id" in store.list_jobs().columns
    assert migrated is not None
    assert migrated.process_id == 0


def test_run_store_migrates_legacy_snapshot_provenance_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "trade_bot.duckdb"
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(
            """
            CREATE TABLE run_snapshots (
                run_id VARCHAR PRIMARY KEY, created_at_utc VARCHAR, status VARCHAR,
                artifact_path VARCHAR, schema_version INTEGER, config_path VARCHAR,
                events_path VARCHAR, macro_path VARCHAR, news_path VARCHAR,
                config_hash VARCHAR, events_hash VARCHAR, macro_hash VARCHAR,
                news_hash VARCHAR, combined_config_hash VARCHAR, refresh_data BOOLEAN,
                refresh_macro BOOLEAN, refresh_news BOOLEAN, market_date VARCHAR,
                risk_status VARCHAR, recommended_action VARCHAR,
                risk_budget_multiplier DOUBLE, price_rows BIGINT, price_columns BIGINT,
                macro_columns BIGINT, strategy_count BIGINT, error_message VARCHAR
            )
            """
        )
    finally:
        connection.close()

    store = RunStore(
        db_path,
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    assert "provenance_json" in store.list_snapshots().columns


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
            auto_prune=False,
        )
        for _ in range(3)
    ]

    dry_run_candidates = store.prune_snapshots(
        keep_latest=1,
        keep_per_market_date=1,
        keep_all_capture_days=0,
        apply=False,
    )
    assert len(dry_run_candidates) == 2
    assert all(Path(manifest.artifact_path).exists() for manifest in manifests)
    assert len(store.list_snapshots(limit=10)) == 3

    applied_candidates = store.prune_snapshots(
        keep_latest=1,
        keep_per_market_date=1,
        keep_all_capture_days=0,
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
    timestamps = iter(f"{market_date.date()}T20:00:00+00:00" for market_date in market_dates)
    monkeypatch.setattr("trade_bot.storage.run_store.utc_now_iso", lambda: next(timestamps))
    manifests = [
        store.save_snapshot(
            _baseline_run(str(market_date.date())),
            config_path=config_path,
            events_path=events_path,
            macro_path=macro_path,
            news_path=news_path,
            auto_prune=False,
        )
        for market_date in market_dates
    ]

    candidates = store.prune_snapshots(
        keep_latest=0,
        keep_per_market_date=1,
        keep_recent_market_days=5,
        keep_all_capture_days=0,
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


def test_run_store_auto_prunes_duplicate_market_date_snapshots(
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
            "2026-07-01T12:00:00+00:00",
            "2026-07-01T20:00:00+00:00",
        ]
    )
    monkeypatch.setattr("trade_bot.storage.run_store.utc_now_iso", lambda: next(timestamps))

    manifests = [
        store.save_snapshot(
            _baseline_run("2026-07-01"),
            config_path=config_path,
            events_path=events_path,
            macro_path=macro_path,
            news_path=news_path,
            prune_keep_all_capture_days=0,
        )
        for _ in range(3)
    ]
    remaining = store.list_snapshots(limit=10)

    assert remaining["run_id"].tolist() == [manifests[-1].run_id]
    assert Path(manifests[-1].artifact_path).exists()
    assert not Path(manifests[0].artifact_path).exists()
    assert not Path(manifests[1].artifact_path).exists()


def test_run_store_auto_prunes_to_recent_daily_and_older_weekly_snapshots(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    config_path, events_path, macro_path, news_path = _config_files(tmp_path)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )
    market_dates = pd.bdate_range("2026-06-01", "2026-07-15")
    timestamps = iter(f"{market_date.date()}T20:00:00+00:00" for market_date in market_dates)
    monkeypatch.setattr("trade_bot.storage.run_store.utc_now_iso", lambda: next(timestamps))

    for market_date in market_dates:
        store.save_snapshot(
            _baseline_run(str(market_date.date())),
            config_path=config_path,
            events_path=events_path,
            macro_path=macro_path,
            news_path=news_path,
            prune_keep_latest=0,
            prune_keep_per_market_date=1,
            prune_keep_recent_market_days=5,
            prune_keep_all_capture_days=0,
            prune_keep_weekly_older=1,
            prune_weekly_frequency="W-WED",
        )

    remaining = store.list_snapshots(limit=100)
    remaining_dates = pd.to_datetime(remaining["market_date"])
    recent_dates = set(market_dates[-5:].date)
    older = remaining_dates[~remaining_dates.dt.date.isin(recent_dates)]

    assert remaining["market_date"].duplicated().sum() == 0
    assert set(remaining_dates.dt.date).issuperset(recent_dates)
    assert older.dt.to_period("W-WED").nunique() == len(older)
    assert len(remaining) == len(recent_dates) + market_dates[:-5].to_period("W-WED").nunique()


def test_run_store_retains_intraday_captures_for_one_week_then_latest_in_day(
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
            "2026-08-10T14:00:00+00:00",
            "2026-08-10T18:00:00+00:00",
            "2026-08-10T22:00:00+00:00",
        ]
    )
    monkeypatch.setattr("trade_bot.storage.run_store.utc_now_iso", lambda: next(timestamps))

    manifests = [
        store.save_snapshot(
            _baseline_run("2026-08-10"),
            config_path=config_path,
            events_path=events_path,
            macro_path=macro_path,
            news_path=news_path,
            auto_prune=False,
        )
        for _ in range(3)
    ]

    fresh_candidates = store.prune_snapshots(
        keep_latest=0,
        keep_per_market_date=1,
        keep_recent_market_days=30,
        keep_all_capture_days=7,
        now_utc="2026-08-17T13:59:59+00:00",
        apply=False,
    )
    assert fresh_candidates.empty

    aged_candidates = store.prune_snapshots(
        keep_latest=0,
        keep_per_market_date=1,
        keep_recent_market_days=30,
        keep_all_capture_days=7,
        now_utc="2026-08-18T22:00:01+00:00",
        apply=True,
    )
    remaining = store.list_snapshots(limit=10)

    assert set(aged_candidates["run_id"]) == {manifests[0].run_id, manifests[1].run_id}
    assert remaining["run_id"].tolist() == [manifests[2].run_id]
    assert remaining.iloc[0]["stored_at_utc"] == "2026-08-10T22:00:00+00:00"


def test_latest_in_day_uses_actual_storage_order_not_historical_as_of_time(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    config_path, events_path, macro_path, news_path = _config_files(tmp_path)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )
    stored_times = iter(
        [
            "2026-08-19T14:00:00+00:00",
            "2026-08-19T18:00:00+00:00",
        ]
    )
    monkeypatch.setattr("trade_bot.storage.run_store.utc_now_iso", lambda: next(stored_times))

    first = store.save_snapshot(
        _baseline_run("2020-03-16"),
        config_path=config_path,
        events_path=events_path,
        macro_path=macro_path,
        news_path=news_path,
        created_at_utc="2020-03-16T22:00:02+00:00",
        auto_prune=False,
    )
    second = store.save_snapshot(
        _baseline_run("2020-03-16"),
        config_path=config_path,
        events_path=events_path,
        macro_path=macro_path,
        news_path=news_path,
        created_at_utc="2020-03-16T22:00:01+00:00",
        auto_prune=False,
    )

    candidates = store.prune_snapshots(
        keep_latest=0,
        keep_per_market_date=1,
        keep_recent_market_days=30,
        keep_all_capture_days=7,
        now_utc="2026-08-27T18:00:01+00:00",
        apply=True,
    )
    remaining = store.list_snapshots(limit=10)

    assert candidates["run_id"].tolist() == [first.run_id]
    assert remaining["run_id"].tolist() == [second.run_id]


def test_run_store_lists_snapshots_by_market_date_before_created_at(tmp_path: Path) -> None:
    config_path, events_path, macro_path, news_path = _config_files(tmp_path)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    old_created_later_market = store.save_snapshot(
        _baseline_run("2026-07-02"),
        config_path=config_path,
        events_path=events_path,
        macro_path=macro_path,
        news_path=news_path,
        created_at_utc="2026-07-02T22:00:00+00:00",
        auto_prune=False,
    )
    new_created_earlier_market = store.save_snapshot(
        _baseline_run("2026-07-01"),
        config_path=config_path,
        events_path=events_path,
        macro_path=macro_path,
        news_path=news_path,
        created_at_utc="2026-07-03T22:00:00+00:00",
        auto_prune=False,
    )

    snapshots = store.list_snapshots(limit=2)

    assert snapshots["run_id"].tolist() == [
        old_created_later_market.run_id,
        new_created_earlier_market.run_id,
    ]


def test_run_store_purges_snapshot_store(tmp_path: Path) -> None:
    config_path, events_path, macro_path, news_path = _config_files(tmp_path)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )
    manifests = [
        store.save_snapshot(
            _baseline_run(f"2026-07-0{day}"),
            config_path=config_path,
            events_path=events_path,
            macro_path=macro_path,
            news_path=news_path,
            created_at_utc=f"2026-07-0{day}T22:00:00+00:00",
            auto_prune=False,
        )
        for day in range(1, 4)
    ]

    dry_run = store.purge_snapshots(apply=False)
    assert len(dry_run) == len(manifests)
    assert dry_run["pruned"].tolist() == [False, False, False]
    assert all(Path(manifest.artifact_path).exists() for manifest in manifests)

    applied = store.purge_snapshots(apply=True)

    assert len(applied) == len(manifests)
    assert applied["pruned"].tolist() == [True, True, True]
    assert store.list_snapshots(limit=10).empty
    assert all(not Path(manifest.artifact_path).exists() for manifest in manifests)


def test_run_store_starts_daily_update_job(tmp_path: Path, monkeypatch: object) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class DummyPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))
            self.pid = os.getpid()

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
    assert jobs.iloc[0]["process_id"] == os.getpid()
    assert "run-daily-update" in str(jobs.iloc[0]["command"])


def test_run_store_reuses_recent_active_duplicate_job(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class DummyPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))
            self.pid = os.getpid()

    monkeypatch.setattr("trade_bot.storage.run_store.subprocess.Popen", DummyPopen)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    first = store.start_daily_update_job(refresh_data=False, refresh_macro=False)
    duplicate = store.start_daily_update_job(refresh_data=False, refresh_macro=False)

    assert duplicate.job_id == first.job_id
    assert len(calls) == 1
    assert len(store.list_jobs(limit=10)) == 1


def test_run_store_reuses_a_fresh_queued_duplicate_during_pid_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class DummyPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))
            self.pid = os.getpid()

    monkeypatch.setattr("trade_bot.storage.run_store.subprocess.Popen", DummyPopen)
    monkeypatch.setattr(RunStore, "_set_job_process_id", lambda *_args: None)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    first = store.start_daily_update_job(refresh_data=False, refresh_macro=False)
    duplicate = store.start_daily_update_job(refresh_data=False, refresh_macro=False)

    assert first.process_id == 0
    assert duplicate.job_id == first.job_id
    assert len(calls) == 1


def test_run_store_fails_dead_duplicate_before_starting_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class DummyPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))
            self.pid = 424_242

    monkeypatch.setattr("trade_bot.storage.run_store.subprocess.Popen", DummyPopen)
    monkeypatch.setattr(RunStore, "_process_is_alive", staticmethod(lambda _pid: False))
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    first = store.start_daily_update_job(refresh_data=False, refresh_macro=False)
    replacement = store.start_daily_update_job(refresh_data=False, refresh_macro=False)
    jobs = store.list_jobs(limit=10).set_index("job_id")

    assert replacement.job_id != first.job_id
    assert len(calls) == 2
    assert jobs.loc[first.job_id, "status"] == "failed"
    assert "no longer running" in jobs.loc[first.job_id, "error_message"]
    assert jobs.loc[replacement.job_id, "status"] == "queued"


def test_run_store_running_duplicate_age_uses_started_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class DummyPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))
            self.pid = os.getpid()

    now = pd.Timestamp.now(tz="UTC").isoformat()
    timestamps = iter(
        [
            "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:00:00+00:00",
            now,
            now,
        ]
    )
    monkeypatch.setattr("trade_bot.storage.run_store.subprocess.Popen", DummyPopen)
    monkeypatch.setattr("trade_bot.storage.run_store.utc_now_iso", lambda: next(timestamps))
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    first = store.start_daily_update_job(refresh_data=False, refresh_macro=False)
    store.mark_job_running(first.job_id)
    duplicate = store.start_daily_update_job(refresh_data=False, refresh_macro=False)

    assert duplicate.job_id == first.job_id
    assert len(calls) == 1


def test_run_store_marks_job_failed_when_process_spawn_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OSError("spawn unavailable")

    monkeypatch.setattr("trade_bot.storage.run_store.subprocess.Popen", BrokenPopen)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    with pytest.raises(OSError, match="spawn unavailable"):
        store.start_daily_update_job(refresh_data=False, refresh_macro=False)

    job = store.list_jobs(limit=1).iloc[0]
    assert job["status"] == "failed"
    assert "could not start: spawn unavailable" in job["error_message"]


def test_run_store_marks_job_failed_when_log_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open

    def failing_log_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.suffix == ".log":
            raise OSError("log unavailable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_log_open)
    store = RunStore(
        tmp_path / "trade_bot.duckdb",
        artifact_dir=tmp_path / "snapshots",
        job_log_dir=tmp_path / "jobs",
    )

    with pytest.raises(OSError, match="log unavailable"):
        store.start_daily_update_job(refresh_data=False, refresh_macro=False)

    job = store.list_jobs(limit=1).iloc[0]
    assert job["status"] == "failed"
    assert "could not start: log unavailable" in job["error_message"]


def test_run_store_starts_targeted_update_jobs(tmp_path: Path, monkeypatch: object) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class DummyPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))
            self.pid = os.getpid()

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
