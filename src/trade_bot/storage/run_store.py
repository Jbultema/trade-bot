from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MACRO_PATH,
    DEFAULT_ML_DIAGNOSTICS_DIR,
    DEFAULT_MONITORING_TOP_N,
    DEFAULT_NEWS_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
)
from trade_bot.research.baselines import BaselineRun

SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SnapshotFingerprint:
    config_hash: str
    events_hash: str
    macro_hash: str
    news_hash: str

    @property
    def combined_hash(self) -> str:
        payload = "|".join(
            [
                self.config_hash,
                self.events_hash,
                self.macro_hash,
                self.news_hash,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SnapshotManifest:
    run_id: str
    created_at_utc: str
    status: str
    artifact_path: str
    schema_version: int
    config_path: str
    events_path: str
    macro_path: str
    news_path: str
    config_hash: str
    events_hash: str
    macro_hash: str
    news_hash: str
    combined_config_hash: str
    refresh_data: bool
    refresh_macro: bool
    refresh_news: bool
    market_date: str
    risk_status: str
    recommended_action: str
    risk_budget_multiplier: float
    price_rows: int
    price_columns: int
    macro_columns: int
    strategy_count: int
    error_message: str = ""


@dataclass(frozen=True)
class SnapshotJob:
    job_id: str
    created_at_utc: str
    started_at_utc: str
    completed_at_utc: str
    status: str
    command: str
    log_path: str
    run_id: str
    error_message: str


class RunStore:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
        artifact_dir: str | Path = DEFAULT_RUN_STORE_ARTIFACT_DIR,
        job_log_dir: str | Path = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.artifact_dir = Path(artifact_dir)
        self.job_log_dir = Path(job_log_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.job_log_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def save_snapshot(
        self,
        run: BaselineRun,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        events_path: str | Path = DEFAULT_EVENTS_PATH,
        macro_path: str | Path = DEFAULT_MACRO_PATH,
        news_path: str | Path = DEFAULT_NEWS_PATH,
        refresh_data: bool = False,
        refresh_macro: bool = False,
        refresh_news: bool = False,
    ) -> SnapshotManifest:
        fingerprint = build_snapshot_fingerprint(config_path, events_path, macro_path, news_path)
        created_at_utc = utc_now_iso()
        run_id = _new_run_id(created_at_utc, fingerprint)
        artifact_path = self.artifact_dir / f"{run_id}.pkl"
        with artifact_path.open("wb") as handle:
            pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)

        trade_summary = _first_row(run.trade_decision.summary)
        manifest = SnapshotManifest(
            run_id=run_id,
            created_at_utc=created_at_utc,
            status="completed",
            artifact_path=str(artifact_path),
            schema_version=SNAPSHOT_SCHEMA_VERSION,
            config_path=str(config_path),
            events_path=str(events_path),
            macro_path=str(macro_path),
            news_path=str(news_path),
            config_hash=fingerprint.config_hash,
            events_hash=fingerprint.events_hash,
            macro_hash=fingerprint.macro_hash,
            news_hash=fingerprint.news_hash,
            combined_config_hash=fingerprint.combined_hash,
            refresh_data=refresh_data,
            refresh_macro=refresh_macro,
            refresh_news=refresh_news,
            market_date=run.current_state.market_date,
            risk_status=run.current_state.risk_status,
            recommended_action=str(trade_summary.get("recommended_action", "")),
            risk_budget_multiplier=_optional_float(
                trade_summary.get("risk_budget_multiplier", 0.0)
            ),
            price_rows=int(run.prices.shape[0]),
            price_columns=int(run.prices.shape[1]),
            macro_columns=int(run.macro_data.shape[1]),
            strategy_count=int(len(run.results)),
        )
        self._upsert_manifest(manifest)
        return manifest

    def load_snapshot(self, run_id: str) -> tuple[BaselineRun, SnapshotManifest]:
        manifest = self.get_snapshot(run_id)
        if manifest is None:
            msg = f"Snapshot not found: {run_id}"
            raise FileNotFoundError(msg)
        artifact_path = Path(manifest.artifact_path)
        with artifact_path.open("rb") as handle:
            run = pickle.load(handle)
        if not isinstance(run, BaselineRun):
            msg = f"Snapshot artifact does not contain a BaselineRun: {artifact_path}"
            raise TypeError(msg)
        return run, manifest

    def load_latest_snapshot(
        self,
        *,
        fingerprint: SnapshotFingerprint | None = None,
        require_matching_config: bool = True,
    ) -> tuple[BaselineRun, SnapshotManifest] | None:
        manifest = self.latest_snapshot(
            fingerprint=fingerprint,
            require_matching_config=require_matching_config,
        )
        if manifest is None:
            return None
        return self.load_snapshot(manifest.run_id)

    def latest_snapshot(
        self,
        *,
        fingerprint: SnapshotFingerprint | None = None,
        require_matching_config: bool = True,
    ) -> SnapshotManifest | None:
        frame = self.list_snapshots(limit=100)
        if frame.empty:
            return None
        if fingerprint is not None and require_matching_config:
            frame = frame[frame["combined_config_hash"] == fingerprint.combined_hash]
        if frame.empty:
            return None
        row = frame.iloc[0].to_dict()
        return _manifest_from_mapping(row)

    def get_snapshot(self, run_id: str) -> SnapshotManifest | None:
        connection = self._connect()
        try:
            frame = connection.execute(
                """
                SELECT *
                FROM run_snapshots
                WHERE run_id = ?
                """,
                [run_id],
            ).fetchdf()
        finally:
            connection.close()
        if frame.empty:
            return None
        return _manifest_from_mapping(frame.iloc[0].to_dict())

    def list_snapshots(self, *, limit: int = 20) -> pd.DataFrame:
        connection = self._connect()
        try:
            return connection.execute(
                """
                SELECT *
                FROM run_snapshots
                WHERE status = 'completed'
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                [limit],
            ).fetchdf()
        finally:
            connection.close()

    def create_job(self, command: list[str], log_path: str | Path) -> SnapshotJob:
        job = SnapshotJob(
            job_id=_new_job_id(),
            created_at_utc=utc_now_iso(),
            started_at_utc="",
            completed_at_utc="",
            status="queued",
            command=" ".join(command),
            log_path=str(log_path),
            run_id="",
            error_message="",
        )
        self._upsert_job(job)
        return job

    def start_snapshot_build_job(
        self,
        *,
        config_path: str | Path,
        events_path: str | Path,
        macro_path: str | Path,
        news_path: str | Path,
        refresh_data: bool = False,
        refresh_macro: bool = False,
        refresh_news: bool = False,
    ) -> SnapshotJob:
        log_path = self.job_log_dir / f"{_new_job_id()}.log"
        command = [
            sys.executable,
            "-m",
            "trade_bot.cli",
            "build-snapshot",
            "--config",
            str(config_path),
            "--events",
            str(events_path),
            "--macro",
            str(macro_path),
            "--news",
            str(news_path),
            "--store",
            str(self.db_path),
            "--artifact-dir",
            str(self.artifact_dir),
            "--job-log-dir",
            str(self.job_log_dir),
        ]
        if refresh_data:
            command.append("--refresh-data")
        if refresh_macro:
            command.append("--refresh-macro")
        if refresh_news:
            command.append("--refresh-news")

        job = self.create_job(command, log_path)
        command.extend(["--job-id", job.job_id])
        self._upsert_job(
            SnapshotJob(
                job_id=job.job_id,
                created_at_utc=job.created_at_utc,
                started_at_utc=job.started_at_utc,
                completed_at_utc=job.completed_at_utc,
                status=job.status,
                command=" ".join(command),
                log_path=job.log_path,
                run_id=job.run_id,
                error_message=job.error_message,
            )
        )

        env = os.environ.copy()
        src_path = str(Path.cwd() / "src")
        env["PYTHONPATH"] = (
            f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
        )
        with Path(job.log_path).open("ab") as log_handle:
            subprocess.Popen(  # noqa: S603
                command,
                cwd=Path.cwd(),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
        )
        return job

    def start_daily_update_job(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        events_path: str | Path = DEFAULT_EVENTS_PATH,
        macro_path: str | Path = DEFAULT_MACRO_PATH,
        news_path: str | Path = DEFAULT_NEWS_PATH,
        report_path: str | Path = DEFAULT_REPORT_PATH,
        experiment_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        refresh_data: bool = True,
        refresh_macro: bool = True,
        refresh_news: bool = True,
        migrate_warehouse: bool = True,
        paper_valuation: bool = True,
    ) -> SnapshotJob:
        log_path = self.job_log_dir / f"{_new_job_id()}.log"
        command = [
            sys.executable,
            "-m",
            "trade_bot.cli",
            "run-daily-update",
            "--config",
            str(config_path),
            "--events",
            str(events_path),
            "--macro",
            str(macro_path),
            "--news",
            str(news_path),
            "--store",
            str(self.db_path),
            "--artifact-dir",
            str(self.artifact_dir),
            "--job-log-dir",
            str(self.job_log_dir),
            "--report-path",
            str(report_path),
            "--experiment-dir",
            str(experiment_dir),
            "--journal",
            str(journal_path),
        ]
        command.append("--refresh-data" if refresh_data else "--cached-data")
        command.append("--refresh-macro" if refresh_macro else "--cached-macro")
        command.append("--refresh-news" if refresh_news else "--cached-news")
        command.append("--migrate-warehouse" if migrate_warehouse else "--skip-warehouse")
        command.append("--paper-valuation" if paper_valuation else "--skip-paper-valuation")

        return self._start_background_job(command, log_path)

    def start_warehouse_migration_job(
        self,
        *,
        experiment_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
    ) -> SnapshotJob:
        log_path = self.job_log_dir / f"{_new_job_id()}.log"
        command = [
            sys.executable,
            "-m",
            "trade_bot.cli",
            "migrate-warehouse",
            "--store",
            str(self.db_path),
            "--artifact-dir",
            str(self.artifact_dir),
            "--job-log-dir",
            str(self.job_log_dir),
            "--experiment-dir",
            str(experiment_dir),
            "--journal",
            str(journal_path),
        ]
        return self._start_background_job(command, log_path)

    def start_paper_valuation_job(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> SnapshotJob:
        log_path = self.job_log_dir / f"{_new_job_id()}.log"
        command = [
            sys.executable,
            "-m",
            "trade_bot.cli",
            "run-paper-valuation",
            "--config",
            str(config_path),
            "--store",
            str(self.db_path),
            "--artifact-dir",
            str(self.artifact_dir),
            "--job-log-dir",
            str(self.job_log_dir),
        ]
        return self._start_background_job(command, log_path)

    def start_monitoring_start_reset_job(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        start_date: str,
        mode: str = "paper",
        account: str | None = None,
        status: str = "active",
        value_after_reset: bool = True,
    ) -> SnapshotJob:
        log_path = self.job_log_dir / f"{_new_job_id()}.log"
        command = [
            sys.executable,
            "-m",
            "trade_bot.cli",
            "reset-monitoring-start-date",
            "--config",
            str(config_path),
            "--store",
            str(self.db_path),
            "--artifact-dir",
            str(self.artifact_dir),
            "--job-log-dir",
            str(self.job_log_dir),
            "--start-date",
            start_date,
            "--mode",
            mode,
            "--status",
            status,
        ]
        if account:
            command.extend(["--account", account])
        command.append("--value-after-reset" if value_after_reset else "--skip-valuation")
        return self._start_background_job(command, log_path)

    def start_monitoring_seed_job(
        self,
        *,
        mode: str = "paper",
        account: str = "default_paper_account",
        capital_base: float = 10_000.0,
        top_n: int = DEFAULT_MONITORING_TOP_N,
        start_date: str | None = None,
    ) -> SnapshotJob:
        log_path = self.job_log_dir / f"{_new_job_id()}.log"
        command = [
            sys.executable,
            "-m",
            "trade_bot.cli",
            "seed-monitoring-windows",
            "--store",
            str(self.db_path),
            "--artifact-dir",
            str(self.artifact_dir),
            "--job-log-dir",
            str(self.job_log_dir),
            "--mode",
            mode,
            "--account",
            account,
            "--capital-base",
            str(capital_base),
            "--top-n",
            str(top_n),
        ]
        if start_date:
            command.extend(["--start-date", start_date])
        return self._start_background_job(command, log_path)

    def start_ml_diagnostics_job(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        output_dir: str | Path = DEFAULT_ML_DIAGNOSTICS_DIR,
        profile: str = "standard",
        refresh_data: bool = False,
        step_days: int | None = None,
    ) -> SnapshotJob:
        log_path = self.job_log_dir / f"{_new_job_id()}.log"
        command = [
            sys.executable,
            "-m",
            "trade_bot.cli",
            "run-ml-diagnostics",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--profile",
            profile,
            "--store",
            str(self.db_path),
            "--artifact-dir",
            str(self.artifact_dir),
            "--job-log-dir",
            str(self.job_log_dir),
        ]
        if refresh_data:
            command.append("--refresh-data")
        if step_days is not None:
            command.extend(["--step-days", str(step_days)])
        return self._start_background_job(command, log_path)

    def _start_background_job(self, command: list[str], log_path: Path) -> SnapshotJob:
        job = self.create_job(command, log_path)
        command.extend(["--job-id", job.job_id])
        self._upsert_job(
            SnapshotJob(
                job_id=job.job_id,
                created_at_utc=job.created_at_utc,
                started_at_utc=job.started_at_utc,
                completed_at_utc=job.completed_at_utc,
                status=job.status,
                command=" ".join(command),
                log_path=job.log_path,
                run_id=job.run_id,
                error_message=job.error_message,
            )
        )

        env = os.environ.copy()
        src_path = str(Path.cwd() / "src")
        env["PYTHONPATH"] = (
            f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
        )
        with Path(job.log_path).open("ab") as log_handle:
            subprocess.Popen(  # noqa: S603
                command,
                cwd=Path.cwd(),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
        return job

    def mark_job_running(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        self._upsert_job(
            SnapshotJob(
                job_id=job.job_id,
                created_at_utc=job.created_at_utc,
                started_at_utc=utc_now_iso(),
                completed_at_utc=job.completed_at_utc,
                status="running",
                command=job.command,
                log_path=job.log_path,
                run_id=job.run_id,
                error_message="",
            )
        )

    def mark_job_completed(self, job_id: str, run_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        self._upsert_job(
            SnapshotJob(
                job_id=job.job_id,
                created_at_utc=job.created_at_utc,
                started_at_utc=job.started_at_utc or utc_now_iso(),
                completed_at_utc=utc_now_iso(),
                status="completed",
                command=job.command,
                log_path=job.log_path,
                run_id=run_id,
                error_message="",
            )
        )

    def mark_job_failed(self, job_id: str, error_message: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        self._upsert_job(
            SnapshotJob(
                job_id=job.job_id,
                created_at_utc=job.created_at_utc,
                started_at_utc=job.started_at_utc or utc_now_iso(),
                completed_at_utc=utc_now_iso(),
                status="failed",
                command=job.command,
                log_path=job.log_path,
                run_id=job.run_id,
                error_message=error_message,
            )
        )

    def get_job(self, job_id: str) -> SnapshotJob | None:
        connection = self._connect()
        try:
            frame = connection.execute(
                """
                SELECT *
                FROM snapshot_jobs
                WHERE job_id = ?
                """,
                [job_id],
            ).fetchdf()
        finally:
            connection.close()
        if frame.empty:
            return None
        return _job_from_mapping(frame.iloc[0].to_dict())

    def list_jobs(self, *, limit: int = 10) -> pd.DataFrame:
        connection = self._connect()
        try:
            return connection.execute(
                """
                SELECT *
                FROM snapshot_jobs
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                [limit],
            ).fetchdf()
        finally:
            connection.close()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_snapshots (
                    run_id VARCHAR PRIMARY KEY,
                    created_at_utc VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    artifact_path VARCHAR NOT NULL,
                    schema_version INTEGER NOT NULL,
                    config_path VARCHAR NOT NULL,
                    events_path VARCHAR NOT NULL,
                    macro_path VARCHAR NOT NULL,
                    news_path VARCHAR NOT NULL,
                    config_hash VARCHAR NOT NULL,
                    events_hash VARCHAR NOT NULL,
                    macro_hash VARCHAR NOT NULL,
                    news_hash VARCHAR NOT NULL,
                    combined_config_hash VARCHAR NOT NULL,
                    refresh_data BOOLEAN NOT NULL,
                    refresh_macro BOOLEAN NOT NULL,
                    refresh_news BOOLEAN NOT NULL,
                    market_date VARCHAR NOT NULL,
                    risk_status VARCHAR NOT NULL,
                    recommended_action VARCHAR NOT NULL,
                    risk_budget_multiplier DOUBLE NOT NULL,
                    price_rows BIGINT NOT NULL,
                    price_columns BIGINT NOT NULL,
                    macro_columns BIGINT NOT NULL,
                    strategy_count BIGINT NOT NULL,
                    error_message VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshot_jobs (
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
        finally:
            connection.close()

    def _upsert_manifest(self, manifest: SnapshotManifest) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO run_snapshots
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    manifest.run_id,
                    manifest.created_at_utc,
                    manifest.status,
                    manifest.artifact_path,
                    manifest.schema_version,
                    manifest.config_path,
                    manifest.events_path,
                    manifest.macro_path,
                    manifest.news_path,
                    manifest.config_hash,
                    manifest.events_hash,
                    manifest.macro_hash,
                    manifest.news_hash,
                    manifest.combined_config_hash,
                    manifest.refresh_data,
                    manifest.refresh_macro,
                    manifest.refresh_news,
                    manifest.market_date,
                    manifest.risk_status,
                    manifest.recommended_action,
                    manifest.risk_budget_multiplier,
                    manifest.price_rows,
                    manifest.price_columns,
                    manifest.macro_columns,
                    manifest.strategy_count,
                    manifest.error_message,
                ],
            )
        finally:
            connection.close()

    def _upsert_job(self, job: SnapshotJob) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO snapshot_jobs
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    job.job_id,
                    job.created_at_utc,
                    job.started_at_utc,
                    job.completed_at_utc,
                    job.status,
                    job.command,
                    job.log_path,
                    job.run_id,
                    job.error_message,
                ],
            )
        finally:
            connection.close()


def build_snapshot_fingerprint(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    macro_path: str | Path = DEFAULT_MACRO_PATH,
    news_path: str | Path = DEFAULT_NEWS_PATH,
) -> SnapshotFingerprint:
    return SnapshotFingerprint(
        config_hash=file_sha256(config_path),
        events_hash=file_sha256(events_path),
        macro_hash=file_sha256(macro_path),
        news_hash=file_sha256(news_path),
    )


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        return f"missing:{target}"
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _new_run_id(created_at_utc: str, fingerprint: SnapshotFingerprint) -> str:
    timestamp = created_at_utc.replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"{timestamp}-{fingerprint.combined_hash[:8]}-{uuid.uuid4().hex[:8]}"


def _new_job_id() -> str:
    timestamp = utc_now_iso().replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"job-{timestamp}-{uuid.uuid4().hex[:8]}"


def _first_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _optional_float(value: object) -> float:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return 0.0
    if numeric != numeric:
        return 0.0
    return numeric


def _manifest_from_mapping(row: dict[str, Any]) -> SnapshotManifest:
    return SnapshotManifest(
        run_id=str(row["run_id"]),
        created_at_utc=str(row["created_at_utc"]),
        status=str(row["status"]),
        artifact_path=str(row["artifact_path"]),
        schema_version=int(row["schema_version"]),
        config_path=str(row["config_path"]),
        events_path=str(row["events_path"]),
        macro_path=str(row["macro_path"]),
        news_path=str(row["news_path"]),
        config_hash=str(row["config_hash"]),
        events_hash=str(row["events_hash"]),
        macro_hash=str(row["macro_hash"]),
        news_hash=str(row["news_hash"]),
        combined_config_hash=str(row["combined_config_hash"]),
        refresh_data=bool(row["refresh_data"]),
        refresh_macro=bool(row["refresh_macro"]),
        refresh_news=bool(row["refresh_news"]),
        market_date=str(row["market_date"]),
        risk_status=str(row["risk_status"]),
        recommended_action=str(row["recommended_action"]),
        risk_budget_multiplier=float(row["risk_budget_multiplier"]),
        price_rows=int(row["price_rows"]),
        price_columns=int(row["price_columns"]),
        macro_columns=int(row["macro_columns"]),
        strategy_count=int(row["strategy_count"]),
        error_message=str(row.get("error_message", "")),
    )


def _job_from_mapping(row: dict[str, Any]) -> SnapshotJob:
    return SnapshotJob(
        job_id=str(row["job_id"]),
        created_at_utc=str(row["created_at_utc"]),
        started_at_utc=str(row["started_at_utc"]),
        completed_at_utc=str(row["completed_at_utc"]),
        status=str(row["status"]),
        command=str(row["command"]),
        log_path=str(row["log_path"]),
        run_id=str(row["run_id"]),
        error_message=str(row["error_message"]),
    )
