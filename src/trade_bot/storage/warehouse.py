from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.config import BotConfig, ExecutionConfig, StrategyConfig, required_strategy_tickers
from trade_bot.DEFAULTS import (
    DEFAULT_EXPERIMENT_REGISTRY_LIMIT,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MONITORING_COHORT_START_DATE,
    DEFAULT_MONITORING_TOP_N,
    DEFAULT_PROSPECTIVE_MONITORING_COHORT,
    DEFAULT_RESET_EXPERIMENTS_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
)
from trade_bot.features.indicators import unusable_required_price_columns
from trade_bot.research.curation import (
    add_research_status,
    default_reference_mask,
    select_curated_strategy_shelf,
)
from trade_bot.research.experiments import ScenarioSizingConfig, apply_scenario_position_sizing
from trade_bot.research.operating_exposure import (
    aggregate_beta_adjusted_spy_delta,
    build_sleeve_exposure_table,
)
from trade_bot.research.validation import add_overfit_diagnostics
from trade_bot.strategies.momentum import build_strategy_weights

WAREHOUSE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WarehouseMigrationResult:
    artifact: str
    rows: int
    table_name: str


@dataclass(frozen=True)
class MonitoringWindowSeedResult:
    window_id: str
    strategy_id: str
    strategy_name: str
    role: str


class TradingWarehouse:
    """Canonical local DuckDB store for operational bot state."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
        *,
        read_only: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    def migrate_experiment_outputs(
        self,
        root: str | Path = DEFAULT_EXPERIMENTS_DIR,
    ) -> list[WarehouseMigrationResult]:
        experiment_root = Path(root)
        artifacts = (
            "candidates",
            "metrics",
            "scorecard",
            "regime_metrics",
            "regime_summary",
            "walk_forward_folds",
            "walk_forward_summary",
            "window_summary",
        )
        migrated_at = utc_now_iso()
        results: list[WarehouseMigrationResult] = []
        for artifact in artifacts:
            frames = []
            for path in sorted(experiment_root.glob(f"iteration_*/{artifact}.csv")):
                frame = pd.read_csv(path)
                frame.insert(0, "iteration", _iteration_from_path(path))
                frame.insert(1, "source_path", str(path))
                frame.insert(2, "migrated_at_utc", migrated_at)
                if "name" in frame.columns and "strategy" not in frame.columns:
                    frame = frame.rename(columns={"name": "strategy"})
                frames.append(frame)
            if not frames:
                continue
            combined = pd.concat(frames, ignore_index=True)
            if artifact == "scorecard":
                combined = add_overfit_diagnostics(combined)
                combined = add_research_status(combined)
            table_name = f"experiment_{artifact}"
            self._replace_table(table_name, combined)
            results.append(
                WarehouseMigrationResult(
                    artifact=artifact,
                    rows=len(combined),
                    table_name=table_name,
                )
            )
        self.refresh_strategy_registry_from_experiments()
        return results

    def migrate_journal_sqlite(
        self,
        path: str | Path = DEFAULT_JOURNAL_PATH,
    ) -> list[WarehouseMigrationResult]:
        journal_path = Path(path)
        if not journal_path.exists():
            return []

        migrated_at = utc_now_iso()
        results: list[WarehouseMigrationResult] = []
        with sqlite3.connect(journal_path) as sqlite_connection:
            table_names = [
                str(row[0])
                for row in sqlite_connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
                ).fetchall()
            ]
            for table_name in table_names:
                frame = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_connection)
                frame.insert(0, "source_path", str(journal_path))
                frame.insert(1, "migrated_at_utc", migrated_at)
                warehouse_table = f"journal_{table_name}"
                self._replace_table(warehouse_table, frame)
                results.append(
                    WarehouseMigrationResult(
                        artifact=f"journal:{table_name}",
                        rows=len(frame),
                        table_name=warehouse_table,
                    )
                )
        return results

    def refresh_strategy_registry_from_experiments(
        self, *, limit: int = DEFAULT_EXPERIMENT_REGISTRY_LIMIT
    ) -> int:
        scorecards = self.read_table("experiment_scorecard")
        if scorecards.empty or "strategy" not in scorecards:
            return 0

        ranked = _rank_strategy_rows(scorecards).head(limit)
        now = utc_now_iso()
        rows = []
        for _, row in ranked.iterrows():
            strategy_name = str(row["strategy"])
            strategy_id = _strategy_id(strategy_name)
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_version": str(row.get("iteration", "")),
                    "strategy_name": strategy_name,
                    "role": str(row.get("role", "unknown")),
                    "status": _strategy_status(row),
                    "source": "experiment_scorecard",
                    "family": str(row.get("family", "unknown")),
                    "benchmark": "SPY",
                    "universe": "",
                    "params_json": "",
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "notes": str(row.get("hypothesis", "")),
                }
            )
        self._upsert_frame("strategy_registry", pd.DataFrame(rows), "strategy_id")
        return len(rows)

    def refresh_strategy_registry_from_snapshot(
        self,
        baseline_run: Any,
        *,
        run_id: str = "latest",
        market_date: str = "",
    ) -> int:
        metrics = baseline_run.metrics.copy()
        if metrics.empty:
            return 0
        metrics = metrics.reset_index()
        if "name" in metrics.columns and "strategy" not in metrics.columns:
            metrics = metrics.rename(columns={"name": "strategy"})
        if "strategy" not in metrics.columns:
            metrics = metrics.rename(columns={metrics.columns[0]: "strategy"})

        now = utc_now_iso()
        rows = []
        for _, row in metrics.iterrows():
            strategy_name = str(row["strategy"])
            cagr = _optional_float(row.get("cagr"))
            calmar = _optional_float(row.get("calmar"))
            max_drawdown = _optional_float(row.get("max_drawdown"))
            notes = (
                f"Snapshot-operable strategy; CAGR={cagr:.2%}"
                if cagr is not None
                else "Snapshot-operable strategy"
            )
            if calmar is not None:
                notes += f"; Calmar={calmar:.2f}"
            if max_drawdown is not None:
                notes += f"; max drawdown={max_drawdown:.2%}"
            rows.append(
                {
                    "strategy_id": _strategy_id(strategy_name),
                    "strategy_version": str(run_id),
                    "strategy_name": strategy_name,
                    "role": "operable",
                    "status": "operable",
                    "source": "latest_snapshot",
                    "family": "baseline_runtime",
                    "benchmark": "SPY",
                    "universe": "configured_runtime",
                    "params_json": "",
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "notes": notes,
                }
            )
        snapshot_metrics = metrics.copy()
        snapshot_metrics.insert(0, "run_id", run_id)
        snapshot_metrics.insert(1, "market_date", market_date)
        snapshot_metrics.insert(2, "updated_at_utc", now)
        self._replace_table("snapshot_strategy_metrics", snapshot_metrics)
        self._upsert_frame("strategy_registry", pd.DataFrame(rows), "strategy_id")
        return len(rows)

    def seed_monitoring_windows_from_registry(
        self,
        *,
        mode: str = "paper",
        account: str = "default_paper_account",
        capital_base: float = 10_000.0,
        top_n: int = DEFAULT_MONITORING_TOP_N,
        start_date: str | None = None,
    ) -> list[MonitoringWindowSeedResult]:
        strategies = self.list_strategy_registry()
        if strategies.empty:
            self.refresh_strategy_registry_from_experiments()
            strategies = self.list_strategy_registry()
        if strategies.empty:
            return []

        candidate_statuses = {"operable", "promoted", "candidate", "evolve"}
        ranked_strategies = self._monitoring_seed_candidates(strategies)
        reference_mask = _reference_candidate_mask(ranked_strategies)
        candidate_pool = ranked_strategies[~reference_mask]
        candidate_rows = candidate_pool[candidate_pool["status"].isin(candidate_statuses)]
        if candidate_rows.empty:
            candidate_rows = candidate_pool if not candidate_pool.empty else ranked_strategies
        candidate_rows = _select_monitoring_rows(candidate_rows, top_n)
        reference_rows = ranked_strategies[
            reference_mask & default_reference_mask(ranked_strategies)
        ]
        if not reference_rows.empty:
            candidate_rows = (
                pd.concat([candidate_rows, reference_rows], ignore_index=True)
                .drop_duplicates("strategy_id", keep="first")
                .reset_index(drop=True)
            )

        existing = self.list_monitoring_windows(status=None)
        existing_keys = set()
        has_active_champion = False
        if not existing.empty:
            existing_keys = set(
                zip(
                    existing["mode"].astype(str),
                    existing["account"].astype(str),
                    existing["strategy_id"].astype(str),
                    existing["status"].astype(str),
                    strict=False,
                )
            )
            active_champions = existing[
                (existing["mode"].astype(str) == mode)
                & (existing["account"].astype(str) == account)
                & (existing["status"].astype(str) == "active")
                & (existing["window_role"].astype(str) == "champion")
            ]
            has_active_champion = not active_champions.empty

        now = utc_now_iso()
        today = _normalize_monitoring_start_date(start_date)
        rows = []
        seeded: list[MonitoringWindowSeedResult] = []
        seeded_champion = False
        for _, row in candidate_rows.iterrows():
            strategy_id = str(row["strategy_id"])
            if (mode, account, strategy_id, "active") in existing_keys:
                continue
            role = str(row.get("role", "unknown"))
            family = str(row.get("family", "unknown"))
            phase = str(row.get("phase", "unknown"))
            strategy_name = str(row.get("strategy_name", ""))
            if (
                role == "reference_portfolio"
                or family == "reference_portfolio"
                or phase == "reference"
                or strategy_name.startswith("i41_ref_")
            ):
                window_role = "reference"
            elif not has_active_champion and not seeded_champion:
                window_role = "champion"
                seeded_champion = True
            else:
                window_role = "challenger"
            window_id = self._unique_monitoring_window_id(
                _monitoring_window_id(mode, account, strategy_id, today)
            )
            rows.append(
                {
                    "window_id": window_id,
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "mode": mode,
                    "account": account,
                    "strategy_id": strategy_id,
                    "strategy_version": str(row.get("strategy_version", "")),
                    "strategy_name": str(row["strategy_name"]),
                    "window_role": window_role,
                    "benchmark": str(row.get("benchmark", "SPY")),
                    "start_date": today,
                    "end_date": "",
                    "status": "active",
                    "capital_base": float(capital_base),
                    "rebalance_cadence": "human_reviewed_daily_to_weekly",
                    "risk_budget": "",
                    "promotion_rule": "Promote only after forward paper edge survives walk-forward, regime, and drawdown checks.",
                    "kill_rule": "Demote on broken thesis, unacceptable drawdown, repeated missed off-ramp, or deteriorating scenario fit.",
                    "notes": f"Seeded from experiment registry as {window_role}; registry role={role}.",
                    "cohort_id": "",
                    "evidence_basis": "reconstructed_historical",
                    "historical_backfill_allowed": True,
                    "strategy_json": "",
                    "execution_json": "",
                }
            )
            seeded.append(
                MonitoringWindowSeedResult(
                    window_id=window_id,
                    strategy_id=strategy_id,
                    strategy_name=str(row["strategy_name"]),
                    role=window_role,
                )
            )
        if rows:
            self._upsert_frame("monitoring_windows", pd.DataFrame(rows), "window_id")
        return seeded

    def seed_prospective_monitoring_cohort(
        self,
        config: BotConfig,
        *,
        start_date: str,
        account: str = "v22_prospective",
        mode: str = "paper",
        capital_base: float = 10_000.0,
        cohort_id: str | None = None,
    ) -> list[MonitoringWindowSeedResult]:
        """Freeze the fixed V2.2 cohort without treating earlier history as forward evidence."""

        start = _normalize_monitoring_start_date(start_date)
        cohort = cohort_id or f"v22-prospective-{start}"
        missing = [
            strategy_name
            for strategy_name, _role in DEFAULT_PROSPECTIVE_MONITORING_COHORT
            if strategy_name not in config.strategies
        ]
        if missing:
            raise ValueError(
                "Prospective cohort cannot be frozen because configured strategies are missing: "
                + ", ".join(missing)
            )

        existing = self.list_monitoring_windows(status=None)
        existing_keys = (
            set(
                zip(
                    existing.get("cohort_id", pd.Series("", index=existing.index)).astype(str),
                    existing["strategy_name"].astype(str),
                    existing["status"].astype(str),
                    strict=False,
                )
            )
            if not existing.empty
            else set()
        )
        now = utc_now_iso()
        execution_json = json.dumps(config.execution.model_dump(mode="json"), sort_keys=True)
        rows: list[dict[str, object]] = []
        registry_rows: list[dict[str, object]] = []
        seeded: list[MonitoringWindowSeedResult] = []
        for strategy_name, role in DEFAULT_PROSPECTIVE_MONITORING_COHORT:
            if (cohort, strategy_name, "active") in existing_keys:
                continue
            strategy = config.strategies[strategy_name]
            strategy_json = json.dumps(strategy.model_dump(mode="json"), sort_keys=True)
            strategy_version = hashlib.sha256(
                f"{strategy_json}\n{execution_json}".encode()
            ).hexdigest()
            strategy_id = _strategy_id(strategy_name)
            window_id = self._unique_monitoring_window_id(
                _monitoring_window_id(mode, account, strategy_id, start)
            )
            rows.append(
                {
                    "window_id": window_id,
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "mode": mode,
                    "account": account,
                    "strategy_id": strategy_id,
                    "strategy_version": strategy_version,
                    "strategy_name": strategy_name,
                    "window_role": role,
                    "benchmark": "SPY",
                    "start_date": start,
                    "end_date": "",
                    "status": "active",
                    "capital_base": float(capital_base),
                    "rebalance_cadence": str(config.execution.rebalance),
                    "risk_budget": "frozen_strategy_policy",
                    "promotion_rule": (
                        "Prospective observations only; reconstructed history cannot satisfy a "
                        "promotion checkpoint."
                    ),
                    "kill_rule": (
                        "Demote on broken thesis, unacceptable drawdown, repeated missed off-ramp, "
                        "or frozen-rule execution failure."
                    ),
                    "notes": (
                        "Fixed V2.2 prospective cohort; no performance before start_date counts as "
                        "forward evidence."
                    ),
                    "cohort_id": cohort,
                    "evidence_basis": "prospective_no_backfill",
                    "historical_backfill_allowed": False,
                    "strategy_json": strategy_json,
                    "execution_json": execution_json,
                }
            )
            registry_rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_version": strategy_version,
                    "strategy_name": strategy_name,
                    "role": role,
                    "status": "operable" if role != "reference" else "reference",
                    "source": "prospective_cohort",
                    "family": "reference_portfolio" if role == "reference" else "i111",
                    "benchmark": "SPY",
                    "universe": "frozen_configured_runtime",
                    "params_json": strategy_json,
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "notes": f"Frozen in prospective cohort {cohort}.",
                }
            )
            seeded.append(
                MonitoringWindowSeedResult(
                    window_id=window_id,
                    strategy_id=strategy_id,
                    strategy_name=strategy_name,
                    role=role,
                )
            )
        if registry_rows:
            self._upsert_frame("strategy_registry", pd.DataFrame(registry_rows), "strategy_id")
        if rows:
            self._upsert_frame("monitoring_windows", pd.DataFrame(rows), "window_id")
        return seeded

    def monitor_strategy(
        self,
        strategy_name: str,
        *,
        role: str = "challenger",
        mode: str = "paper",
        account: str = "default_paper_account",
        capital_base: float = 10_000.0,
        start_date: str | None = None,
        demote_other_champions: bool = False,
    ) -> MonitoringWindowSeedResult:
        _validate_window_role(role)
        strategies = self.list_strategy_registry()
        if strategies.empty:
            self.refresh_strategy_registry_from_experiments()
            strategies = self.list_strategy_registry()
        if strategies.empty:
            raise ValueError("No strategies are registered. Run migrate-warehouse first.")

        match = strategies[
            (strategies["strategy_name"].astype(str) == strategy_name)
            | (strategies["strategy_id"].astype(str) == strategy_name)
        ]
        if match.empty:
            raise ValueError(f"Strategy is not registered: {strategy_name}")
        row = match.iloc[0]
        strategy_id = str(row["strategy_id"])
        requested_start_date = _normalize_monitoring_start_date(start_date)

        existing = self.list_monitoring_windows(status=None)
        if not existing.empty:
            existing_active = existing[
                (existing["mode"].astype(str) == mode)
                & (existing["account"].astype(str) == account)
                & (existing["strategy_id"].astype(str) == strategy_id)
                & (existing["status"].astype(str) == "active")
            ]
            if start_date is not None:
                existing_active = existing_active[
                    existing_active["start_date"].astype(str) == requested_start_date
                ]
            if not existing_active.empty:
                window = existing_active.sort_values(["start_date", "created_at_utc"]).iloc[-1]
                self.update_monitoring_window(
                    str(window["window_id"]),
                    role=role,
                    status="active",
                    capital_base=capital_base,
                    start_date=start_date,
                    demote_other_champions=demote_other_champions,
                )
                return MonitoringWindowSeedResult(
                    window_id=str(window["window_id"]),
                    strategy_id=strategy_id,
                    strategy_name=str(row["strategy_name"]),
                    role=role,
                )

        today = requested_start_date
        window_id = self._unique_monitoring_window_id(
            _monitoring_window_id(mode, account, strategy_id, today)
        )
        now = utc_now_iso()
        record = pd.DataFrame(
            [
                {
                    "window_id": window_id,
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "mode": mode,
                    "account": account,
                    "strategy_id": strategy_id,
                    "strategy_version": str(row.get("strategy_version", "")),
                    "strategy_name": str(row["strategy_name"]),
                    "window_role": role,
                    "benchmark": str(row.get("benchmark", "SPY")),
                    "start_date": today,
                    "end_date": "",
                    "status": "active",
                    "capital_base": float(capital_base),
                    "rebalance_cadence": "human_reviewed_daily_to_weekly",
                    "risk_budget": "",
                    "promotion_rule": "Promote only after forward paper edge survives walk-forward, regime, and drawdown checks.",
                    "kill_rule": "Demote on broken thesis, unacceptable drawdown, repeated missed off-ramp, or deteriorating scenario fit.",
                    "notes": f"Manually monitored from dashboard/CLI as {role}.",
                    "cohort_id": "",
                    "evidence_basis": "reconstructed_historical",
                    "historical_backfill_allowed": True,
                    "strategy_json": "",
                    "execution_json": "",
                }
            ]
        )
        self._upsert_frame("monitoring_windows", record, "window_id")
        if role == "champion" and demote_other_champions:
            self._demote_other_champions(window_id, mode, account)
        return MonitoringWindowSeedResult(
            window_id=window_id,
            strategy_id=strategy_id,
            strategy_name=str(row["strategy_name"]),
            role=role,
        )

    def update_monitoring_window(
        self,
        window_id: str,
        *,
        role: str | None = None,
        status: str | None = None,
        capital_base: float | None = None,
        start_date: str | None = None,
        demote_other_champions: bool = False,
    ) -> bool:
        windows = self.list_monitoring_windows(status=None)
        if windows.empty:
            return False
        selected = windows[windows["window_id"].astype(str) == window_id]
        if selected.empty:
            return False

        row = selected.iloc[0]
        updates: list[str] = ["updated_at_utc = ?"]
        params: list[object] = [utc_now_iso()]
        if role is not None:
            _validate_window_role(role)
            updates.append("window_role = ?")
            params.append(role)
        if status is not None:
            _validate_window_status(status)
            updates.append("status = ?")
            params.append(status)
            if status != "active" and not str(row.get("end_date", "")):
                updates.append("end_date = ?")
                params.append(date.today().isoformat())
        if capital_base is not None:
            if capital_base <= 0:
                raise ValueError("capital_base must be positive.")
            updates.append("capital_base = ?")
            params.append(float(capital_base))
        normalized_start_date = (
            _normalize_monitoring_start_date(start_date) if start_date is not None else None
        )
        if normalized_start_date is not None:
            updates.append("start_date = ?")
            params.append(normalized_start_date)

        params.append(window_id)
        self._execute(
            f"UPDATE monitoring_windows SET {', '.join(updates)} WHERE window_id = ?",
            params,
        )
        if normalized_start_date is not None and normalized_start_date != str(row["start_date"]):
            self.clear_monitoring_valuations([window_id])
        if role == "champion" and demote_other_champions:
            self._demote_other_champions(
                window_id,
                str(row["mode"]),
                str(row["account"]),
            )
        return True

    def reset_monitoring_start_dates(
        self,
        *,
        start_date: str = DEFAULT_MONITORING_COHORT_START_DATE,
        mode: str | None = "paper",
        account: str | None = None,
        status: str | None = "active",
        clear_valuations: bool = True,
    ) -> int:
        normalized_start_date = _normalize_monitoring_start_date(start_date)
        windows = self.list_monitoring_windows(status=status)
        if windows.empty:
            return 0
        mask = pd.Series(True, index=windows.index)
        if mode is not None:
            mask &= windows["mode"].astype(str) == mode
        if account:
            mask &= windows["account"].astype(str) == account
        selected = windows[mask].copy()
        if selected.empty:
            return 0

        window_ids = selected["window_id"].astype(str).tolist()
        connection = self._connect()
        try:
            update_ids = pd.DataFrame({"window_id": window_ids})
            connection.register("update_window_ids", update_ids)
            connection.execute(
                """
                UPDATE monitoring_windows
                SET start_date = ?, updated_at_utc = ?
                WHERE window_id IN (SELECT window_id FROM update_window_ids)
                """,
                [normalized_start_date, utc_now_iso()],
            )
        finally:
            connection.close()
        if clear_valuations:
            self.clear_monitoring_valuations(window_ids)
        return len(window_ids)

    def clear_monitoring_valuations(self, window_ids: Iterable[str]) -> int:
        window_id_list = [str(window_id) for window_id in window_ids if str(window_id)]
        if not window_id_list or not self.table_exists("strategy_daily_valuations"):
            return 0
        before = self._query("SELECT COUNT(*) AS rows FROM strategy_daily_valuations")["rows"].iloc[
            0
        ]
        connection = self._connect()
        try:
            delete_ids = pd.DataFrame({"window_id": window_id_list})
            connection.register("delete_window_ids", delete_ids)
            connection.execute(
                """
                DELETE FROM strategy_daily_valuations
                WHERE window_id IN (SELECT window_id FROM delete_window_ids)
                """
            )
        finally:
            connection.close()
        after = self._query("SELECT COUNT(*) AS rows FROM strategy_daily_valuations")["rows"].iloc[
            0
        ]
        return int(before - after)

    def _demote_other_champions(self, window_id: str, mode: str, account: str) -> None:
        self._execute(
            """
            UPDATE monitoring_windows
            SET window_role = 'challenger', updated_at_utc = ?
            WHERE window_id <> ?
              AND mode = ?
              AND account = ?
              AND status = 'active'
              AND window_role = 'champion'
            """,
            [utc_now_iso(), window_id, mode, account],
        )

    def _unique_monitoring_window_id(self, base_window_id: str) -> str:
        existing = self.list_monitoring_windows(status=None)
        if existing.empty or base_window_id not in set(existing["window_id"].astype(str)):
            return base_window_id
        existing_ids = set(existing["window_id"].astype(str))
        suffix = 2
        while f"{base_window_id}-{suffix}" in existing_ids:
            suffix += 1
        return f"{base_window_id}-{suffix}"

    def _monitoring_seed_candidates(self, strategies: pd.DataFrame) -> pd.DataFrame:
        scorecards = self.read_table("experiment_scorecard")
        snapshot_metrics = self.read_table("snapshot_strategy_metrics")
        merged = strategies.copy()
        if not snapshot_metrics.empty and "strategy" in snapshot_metrics:
            snapshot_scores = snapshot_metrics.rename(
                columns={
                    "cagr": "snapshot_cagr",
                    "sharpe": "snapshot_sharpe",
                    "max_drawdown": "snapshot_max_drawdown",
                    "calmar": "snapshot_calmar",
                    "average_turnover": "snapshot_average_turnover",
                }
            )
            snapshot_columns = [
                "strategy",
                "snapshot_cagr",
                "snapshot_sharpe",
                "snapshot_max_drawdown",
                "snapshot_calmar",
                "snapshot_average_turnover",
            ]
            merged = merged.merge(
                snapshot_scores[
                    [column for column in snapshot_columns if column in snapshot_scores]
                ],
                left_on="strategy_name",
                right_on="strategy",
                how="left",
            ).drop(columns=["strategy"], errors="ignore")
        if scorecards.empty or "strategy" not in scorecards:
            scorecards = pd.DataFrame()
        latest_scores = (
            _rank_strategy_rows(scorecards).drop_duplicates("strategy", keep="first")
            if not scorecards.empty
            else pd.DataFrame()
        )
        score_columns = [
            "strategy",
            "iteration",
            "phase",
            "promotion_decision",
            "promotion_score",
            "selection_adjusted_promotion_score",
            "growth_constrained_utility_score",
            "growth_utility_tier",
            "tax_model_status",
            "tax_account_type",
            "after_tax_cagr",
            "after_tax_max_drawdown",
            "after_tax_calmar",
            "tax_drag_bps_per_year",
            "after_tax_growth_constrained_utility_score",
            "after_tax_growth_utility_tier",
            "after_tax_terminal_wealth_with_contributions_15y",
            "terminal_wealth_with_contributions_15y",
            "wealth_multiple_vs_spy",
            "wealth_multiple_vs_qqq",
            "drawdown_recovery_return",
            "drawdown_soft_penalty",
            "drawdown_hard_penalty",
            "robustness_score",
            "overfit_risk_score",
            "overfit_risk_label",
            "validation_tier",
            "cagr",
            "sharpe",
            "max_drawdown",
            "calmar",
            "average_turnover",
            "walk_forward_median_cagr",
            "walk_forward_worst_cagr",
            "walk_forward_positive_rate",
            "left_tail_regime_return",
            "research_status",
            "prune_reason",
            "operability_label",
        ]
        if not latest_scores.empty:
            merged = merged.merge(
                latest_scores[[column for column in score_columns if column in latest_scores]],
                left_on="strategy_name",
                right_on="strategy",
                how="left",
            ).drop(columns=["strategy"], errors="ignore")
        merged["status_rank"] = (
            merged["status"]
            .map({"operable": 0, "promoted": 1, "candidate": 2, "evolve": 3, "rejected": 4})
            .fillna(5)
        )
        merged["validation_rank"] = (
            merged.get(
                "validation_tier",
                pd.Series("needs_more_holdout_evidence", index=merged.index),
            )
            .map(
                {
                    "paper_champion_candidate": 0,
                    "paper_challenger_candidate": 1,
                    "needs_more_holdout_evidence": 2,
                    "reject_or_redesign": 3,
                }
            )
            .fillna(2)
        )
        for column in [
            "growth_constrained_utility_score",
            "selection_adjusted_promotion_score",
            "promotion_score",
            "snapshot_calmar",
        ]:
            if column not in merged:
                merged[column] = float("nan")
        merged["snapshot_ready_rank"] = merged["snapshot_calmar"].isna().astype(int)
        merged["monitoring_sort_score"] = (
            merged["snapshot_calmar"]
            .fillna(merged["growth_constrained_utility_score"])
            .fillna(merged["selection_adjusted_promotion_score"])
            .fillna(merged["promotion_score"])
        )
        return merged.sort_values(
            [
                "status_rank",
                "snapshot_ready_rank",
                "monitoring_sort_score",
                "validation_rank",
                "selection_adjusted_promotion_score",
                "promotion_score",
                "strategy_name",
            ],
            ascending=[True, True, False, True, False, False, True],
            na_position="last",
        )

    def top_monitoring_candidates(self, *, limit: int = DEFAULT_MONITORING_TOP_N) -> pd.DataFrame:
        strategies = self.list_strategy_registry()
        if strategies.empty:
            self.refresh_strategy_registry_from_experiments()
            strategies = self.list_strategy_registry()
        if strategies.empty:
            return pd.DataFrame()

        candidate_statuses = {"operable", "promoted", "candidate", "evolve"}
        ranked = self._monitoring_seed_candidates(strategies)
        reference_mask = _reference_candidate_mask(ranked)
        research_status = ranked.get("research_status", pd.Series("", index=ranked.index))
        candidates = ranked[
            ranked["status"].isin(candidate_statuses)
            & ~reference_mask
            & ~research_status.astype(str).eq("pruned_dead_end")
        ].copy()
        if candidates.empty:
            candidates = ranked[~reference_mask].copy()
        candidates = _select_monitoring_rows(candidates, limit).reset_index(drop=True)
        candidates.insert(0, "rank", range(1, len(candidates) + 1))
        return self._enrich_monitoring_candidates(candidates)

    def reference_monitoring_candidates(self) -> pd.DataFrame:
        strategies = self.list_strategy_registry()
        if strategies.empty:
            self.refresh_strategy_registry_from_experiments()
            strategies = self.list_strategy_registry()
        if strategies.empty:
            return pd.DataFrame()

        ranked = self._monitoring_seed_candidates(strategies)
        reference_mask = _reference_candidate_mask(ranked) & default_reference_mask(ranked)
        references = ranked[reference_mask].copy()
        if references.empty:
            return pd.DataFrame()
        references = references.sort_values(
            [
                "promotion_score",
                "selection_adjusted_promotion_score",
                "calmar",
                "strategy_name",
            ],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        references.insert(0, "rank", range(1, len(references) + 1))
        return self._enrich_monitoring_candidates(references)

    def _enrich_monitoring_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        candidates = candidates.copy()
        windows = self.list_monitoring_windows(status="active")
        if not windows.empty:
            window_columns = [
                "strategy_id",
                "window_id",
                "window_role",
                "status",
                "start_date",
            ]
            window_state = (
                windows[[column for column in window_columns if column in windows.columns]]
                .sort_values(["strategy_id", "start_date", "window_id"])
                .drop_duplicates("strategy_id", keep="last")
                .rename(columns={"status": "window_status"})
            )
            candidates = candidates.merge(window_state, on="strategy_id", how="left")
        else:
            for column in ["window_id", "window_role", "window_status", "start_date"]:
                candidates[column] = ""

        valuations = self.read_table("strategy_daily_valuations")
        if not valuations.empty and "window_id" in valuations and "window_id" in candidates:
            valuation_columns = [
                "window_id",
                "valuation_date",
                "equity",
                "cumulative_return",
                "drawdown",
                "excess_return",
            ]
            latest_valuations = (
                valuations[[column for column in valuation_columns if column in valuations.columns]]
                .sort_values(["window_id", "valuation_date"])
                .drop_duplicates("window_id", keep="last")
            )
            candidates = candidates.merge(latest_valuations, on="window_id", how="left")

        snapshot_metrics = self.read_table("snapshot_strategy_metrics")
        operable_names = (
            set(snapshot_metrics["strategy"].astype(str))
            if not snapshot_metrics.empty and "strategy" in snapshot_metrics
            else set()
        )
        candidate_manifests = self.read_table("experiment_candidates")
        candidate_names = (
            set(candidate_manifests["strategy"].astype(str))
            if not candidate_manifests.empty and "strategy" in candidate_manifests
            else set()
        )
        candidates["snapshot_valuation_ready"] = (
            candidates["strategy_name"].astype(str).isin(operable_names)
            | candidates["strategy_name"].astype(str).isin(candidate_names)
            | (candidates["source"].astype(str) == "latest_snapshot")
        )
        window_id = _clean_identifier_series(
            candidates.get("window_id", pd.Series("", index=candidates.index))
        )
        candidates["is_active_window"] = window_id.str.len() > 0
        candidates["is_valued"] = (
            _clean_identifier_series(candidates["valuation_date"]).str.len() > 0
            if "valuation_date" in candidates
            else False
        )
        candidates["monitoring_state"] = candidates.apply(_monitoring_state, axis=1)
        return candidates

    def save_daily_valuations_from_snapshot(
        self,
        baseline_run: Any,
        *,
        market_date: str | None = None,
        execution: ExecutionConfig | None = None,
    ) -> int:
        windows = self.list_monitoring_windows(status="active")
        if windows.empty:
            return 0

        valuation_date = market_date or str(getattr(baseline_run.current_state, "market_date", ""))
        benchmark_equity = _benchmark_equity(baseline_run)
        runtime_results = self._monitoring_runtime_results(
            baseline_run,
            windows,
            execution=execution,
        )
        prospective_results = self._prospective_monitoring_results(baseline_run, windows)
        rows = []
        for _, window in windows.iterrows():
            strategy_name = str(window["strategy_name"])
            result = prospective_results.get(str(window["window_id"]))
            if result is None:
                result = runtime_results.get(strategy_name)
            if result is None:
                continue
            if result.equity.empty:
                continue
            equity_series = result.equity.dropna()
            returns = result.returns.dropna()
            if equity_series.empty:
                continue
            capital_base = float(window["capital_base"])
            start_date = _normalize_monitoring_start_date(str(window.get("start_date", "")))
            strategy_path = _strategy_path_for_monitoring_window(
                equity_series,
                start_date=start_date,
                valuation_date=valuation_date,
            )
            if strategy_path.empty:
                continue
            strategy_start = float(strategy_path.iloc[0])
            strategy_end = float(strategy_path.iloc[-1])
            if strategy_start <= 0:
                continue
            cumulative_return = strategy_end / strategy_start - 1.0
            paper_equity = capital_base * (1.0 + cumulative_return)
            daily_return = _series_return_on_or_before(returns, valuation_date)
            benchmark_return = 0.0
            benchmark_cumulative_return = float("nan")
            benchmark_equity_value = float("nan")
            if benchmark_equity is not None and not benchmark_equity.empty:
                benchmark_path = _strategy_path_for_monitoring_window(
                    benchmark_equity,
                    start_date=start_date,
                    valuation_date=valuation_date,
                )
                if not benchmark_path.empty and float(benchmark_path.iloc[0]) > 0:
                    benchmark_cumulative_return = (
                        float(benchmark_path.iloc[-1]) / float(benchmark_path.iloc[0]) - 1.0
                    )
                    benchmark_equity_value = capital_base * (1.0 + benchmark_cumulative_return)
                    benchmark_return = _series_return_on_or_before(
                        benchmark_equity.pct_change().dropna(),
                        valuation_date,
                    )
            drawdown = _latest_drawdown(strategy_path)
            latest_weights = _latest_strategy_weights(result)
            exposure_diagnostics = _latest_exposure_diagnostics(
                getattr(baseline_run, "prices", pd.DataFrame()),
                latest_weights,
            )
            rows.append(
                {
                    "valuation_id": f"{window['window_id']}:{valuation_date}",
                    "window_id": str(window["window_id"]),
                    "valuation_date": valuation_date,
                    "created_at_utc": utc_now_iso(),
                    "strategy_id": str(window["strategy_id"]),
                    "strategy_version": str(window["strategy_version"]),
                    "strategy_name": strategy_name,
                    "mode": str(window["mode"]),
                    "account": str(window["account"]),
                    "cohort_id": str(window.get("cohort_id", "") or ""),
                    "evidence_basis": str(
                        window.get("evidence_basis", "reconstructed_historical")
                        or "reconstructed_historical"
                    ),
                    "historical_backfill_allowed": bool(
                        _monitoring_backfill_allowed(window)
                    ),
                    "equity": paper_equity,
                    "cash": 0.0,
                    "gross_exposure": _latest_gross_exposure(result),
                    "net_exposure": _latest_net_exposure(result),
                    "daily_return": daily_return,
                    "cumulative_return": cumulative_return,
                    "drawdown": drawdown,
                    "benchmark_equity": benchmark_equity_value,
                    "benchmark_return": benchmark_return,
                    "benchmark_cumulative_return": benchmark_cumulative_return,
                    "excess_return": (
                        cumulative_return - benchmark_cumulative_return
                        if benchmark_cumulative_return == benchmark_cumulative_return
                        else float("nan")
                    ),
                    **exposure_diagnostics,
                    "notes": _monitoring_valuation_note(window),
                }
            )
        if not rows:
            return 0
        self._upsert_frame("strategy_daily_valuations", pd.DataFrame(rows), "valuation_id")
        return len(rows)

    def save_simulation_validation_run(
        self,
        *,
        snapshot_run_id: str,
        market_date: str,
        strategy: str,
        reference_strategies: str,
        horizons: str,
        origin_frequency: str,
        min_train_days: int,
        paths: int,
        block_days: int,
        interval_low: float,
        interval_high: float,
        scenario_history_path: str,
        validation_output_path: str,
        ablation_output_path: str,
        rank_output_path: str,
        validation_summary: dict[str, object],
        validation: pd.DataFrame,
        horizon_summary: pd.DataFrame | None = None,
        ablation_summary: pd.DataFrame | None = None,
    ) -> str:
        validation_run_id = _new_validation_run_id()
        created_at_utc = utc_now_iso()
        run_record = pd.DataFrame(
            [
                {
                    "validation_run_id": validation_run_id,
                    "created_at_utc": created_at_utc,
                    "snapshot_run_id": snapshot_run_id,
                    "market_date": market_date,
                    "strategy": strategy,
                    "reference_strategies": reference_strategies,
                    "horizons": horizons,
                    "origin_frequency": origin_frequency,
                    "min_train_days": int(min_train_days),
                    "paths": int(paths),
                    "block_days": int(block_days),
                    "interval_low": float(interval_low),
                    "interval_high": float(interval_high),
                    "target_interval_coverage": float(interval_high - interval_low),
                    "scenario_history_path": scenario_history_path,
                    "validation_output_path": validation_output_path,
                    "ablation_output_path": ablation_output_path,
                    "rank_output_path": rank_output_path,
                    "primary_validity_read": str(validation_summary.get("validity_read", "")),
                    "primary_distribution_calibration_read": str(
                        validation_summary.get(
                            "distribution_calibration_read",
                            validation_summary.get("validity_read", ""),
                        )
                    ),
                    "primary_action_readiness_read": str(
                        validation_summary.get("action_readiness_read", "")
                    ),
                    "primary_interval_coverage": _optional_float(
                        validation_summary.get("interval_coverage")
                    ),
                    "primary_coverage_error": _optional_float(
                        validation_summary.get("coverage_error")
                    ),
                    "primary_median_abs_error": _optional_float(
                        validation_summary.get("median_abs_error")
                    ),
                    "primary_launch_decision_accuracy": _optional_float(
                        validation_summary.get("launch_decision_accuracy")
                    ),
                }
            ]
        )
        self._upsert_frame("simulation_validation_runs", run_record, "validation_run_id")

        metric_rows = [
            _simulation_validation_summary_metric_row(
                validation_run_id=validation_run_id,
                created_at_utc=created_at_utc,
                strategy=strategy,
                metric_scope="primary_summary",
                variant="current_engine",
                label="Current simulation engine",
                summary=validation_summary,
            )
        ]
        metric_rows.extend(
            _simulation_validation_origin_metric_rows(
                validation_run_id=validation_run_id,
                created_at_utc=created_at_utc,
                strategy=strategy,
                validation=validation,
            )
        )
        if horizon_summary is not None and not horizon_summary.empty:
            metric_rows.extend(
                _simulation_horizon_summary_metric_rows(
                    validation_run_id=validation_run_id,
                    created_at_utc=created_at_utc,
                    strategy=strategy,
                    horizon_summary=horizon_summary,
                )
            )
        if ablation_summary is not None and not ablation_summary.empty:
            metric_rows.extend(
                _simulation_ablation_metric_rows(
                    validation_run_id=validation_run_id,
                    created_at_utc=created_at_utc,
                    strategy=strategy,
                    ablation_summary=ablation_summary,
                )
            )
        self._upsert_frame(
            "simulation_validation_metrics",
            pd.DataFrame(metric_rows),
            "metric_id",
        )
        return validation_run_id

    def simulation_validation_runs(self, *, limit: int = 25) -> pd.DataFrame:
        if not self.table_exists("simulation_validation_runs"):
            return pd.DataFrame()
        return self._query(
            """
            SELECT *
            FROM simulation_validation_runs
            ORDER BY created_at_utc DESC
            LIMIT ?
            """,
            [limit],
        )

    def simulation_validation_metrics(
        self,
        *,
        validation_run_id: str | None = None,
        strategy: str | None = None,
        metric_scope: str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        if not self.table_exists("simulation_validation_metrics"):
            return pd.DataFrame()
        filters = []
        params: list[object] = []
        if validation_run_id:
            filters.append("validation_run_id = ?")
            params.append(validation_run_id)
        if strategy:
            filters.append("strategy = ?")
            params.append(strategy)
        if metric_scope:
            filters.append("metric_scope = ?")
            params.append(metric_scope)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        params.append(limit)
        return self._query(
            f"""
            SELECT *
            FROM simulation_validation_metrics
            {where_clause}
            ORDER BY created_at_utc DESC, metric_scope, variant, origin_date, horizon
            LIMIT ?
            """,
            params,
        )

    def save_cycle_tracker_run(
        self,
        *,
        snapshot_run_id: str,
        market_date: str,
        output_dir: str,
        horizons: str,
        min_train_days: int,
        origin_step_days: int,
        phase_probabilities: pd.DataFrame,
        transition_forecast: pd.DataFrame,
        evidence: pd.DataFrame,
        candidate_scores: pd.DataFrame,
        phase_candidate_frontier: pd.DataFrame,
        validation_metrics: pd.DataFrame,
        path_validation_metrics: pd.DataFrame | None = None,
        path_state_history: pd.DataFrame | None = None,
        path_transition_forecast: pd.DataFrame | None = None,
        phase_reliability: pd.DataFrame | None = None,
        path_reliability: pd.DataFrame | None = None,
        crisis_playback: pd.DataFrame | None = None,
        readout: str,
    ) -> str:
        cycle_run_id = _new_cycle_run_id()
        created_at_utc = utc_now_iso()
        dominant_phase = ""
        dominant_probability = float("nan")
        if not phase_probabilities.empty and "probability" in phase_probabilities:
            top_row = phase_probabilities.sort_values("probability", ascending=False).iloc[0]
            dominant_phase = str(top_row.get("phase", ""))
            dominant_probability = _optional_float(top_row.get("probability")) or float("nan")
        run_record = pd.DataFrame(
            [
                {
                    "cycle_run_id": cycle_run_id,
                    "created_at_utc": created_at_utc,
                    "snapshot_run_id": snapshot_run_id,
                    "market_date": market_date,
                    "output_dir": output_dir,
                    "horizons": horizons,
                    "min_train_days": int(min_train_days),
                    "origin_step_days": int(origin_step_days),
                    "dominant_phase": dominant_phase,
                    "dominant_phase_probability": dominant_probability,
                    "candidate_rows": int(len(candidate_scores)),
                    "frontier_rows": int(len(phase_candidate_frontier)),
                    "validation_rows": int(len(validation_metrics)),
                    "reliability_rows": (
                        int(len(phase_reliability)) if phase_reliability is not None else 0
                    ),
                    "crisis_rows": int(len(crisis_playback)) if crisis_playback is not None else 0,
                    "readout": readout,
                }
            ]
        )
        self._upsert_frame("cycle_tracker_runs", run_record, "cycle_run_id")

        if not phase_probabilities.empty:
            frame = phase_probabilities.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "phase", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_phase_probabilities", frame, "metric_id")
        if not transition_forecast.empty:
            frame = transition_forecast.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "forecast", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_transition_forecast", frame, "metric_id")
        if not evidence.empty:
            frame = evidence.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "evidence", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_evidence", frame, "metric_id")
        if path_state_history is not None and not path_state_history.empty:
            frame = path_state_history.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "pathstate", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_path_state_history", frame, "metric_id")
        if path_transition_forecast is not None and not path_transition_forecast.empty:
            frame = path_transition_forecast.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "pathforecast", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_path_transition_forecast", frame, "metric_id")
        if not candidate_scores.empty:
            frame = candidate_scores.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "candidate", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_candidate_scores", frame, "metric_id")
        if not phase_candidate_frontier.empty:
            frame = phase_candidate_frontier.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "frontier", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_phase_candidate_frontier", frame, "metric_id")
        if not validation_metrics.empty:
            frame = validation_metrics.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "validation", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_validation_metrics", frame, "metric_id")
        if path_validation_metrics is not None and not path_validation_metrics.empty:
            frame = path_validation_metrics.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "pathvalidation", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_path_validation_metrics", frame, "metric_id")
        if phase_reliability is not None and not phase_reliability.empty:
            frame = phase_reliability.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "reliability", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_phase_reliability", frame, "metric_id")
        if path_reliability is not None and not path_reliability.empty:
            frame = path_reliability.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "pathreliability", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_path_reliability", frame, "metric_id")
        if crisis_playback is not None and not crisis_playback.empty:
            frame = crisis_playback.copy()
            frame.insert(0, "metric_id", _metric_ids(cycle_run_id, "crisis", len(frame)))
            frame.insert(1, "cycle_run_id", cycle_run_id)
            frame.insert(2, "created_at_utc", created_at_utc)
            self._upsert_frame("cycle_tracker_crisis_playback", frame, "metric_id")
        return cycle_run_id

    def cycle_tracker_runs(self, *, limit: int = 10) -> pd.DataFrame:
        if not self.table_exists("cycle_tracker_runs"):
            return pd.DataFrame()
        return self._query(
            """
            SELECT *
            FROM cycle_tracker_runs
            ORDER BY created_at_utc DESC
            LIMIT ?
            """,
            [limit],
        )

    def _monitoring_runtime_results(
        self,
        baseline_run: Any,
        windows: pd.DataFrame,
        *,
        execution: ExecutionConfig | None,
    ) -> dict[str, BacktestResult]:
        runtime_results: dict[str, BacktestResult] = dict(getattr(baseline_run, "results", {}))
        if execution is None:
            return runtime_results

        missing_names = sorted(set(windows["strategy_name"].astype(str)) - set(runtime_results))
        if not missing_names:
            return runtime_results

        candidate_manifests = self._candidate_manifests_for_strategy_names(missing_names)
        if candidate_manifests.empty or "strategy_json" not in candidate_manifests:
            return runtime_results
        manifest_rows = (
            candidate_manifests[candidate_manifests["strategy"].astype(str).isin(missing_names)]
            .sort_values(["strategy", "iteration"])
            .drop_duplicates("strategy", keep="last")
        )
        prices = getattr(baseline_run, "prices", pd.DataFrame())
        if prices.empty:
            return runtime_results

        for _, row in manifest_rows.iterrows():
            strategy_name = str(row["strategy"])
            strategy = _strategy_from_candidate_manifest(row)
            if strategy is None:
                continue
            strategy_prices = _candidate_strategy_prices(
                prices,
                strategy,
            )
            if strategy_prices.empty:
                continue
            target_weights = build_strategy_weights(strategy_prices, strategy)
            scenario_sizing = _scenario_sizing_from_candidate_manifest(row)
            if scenario_sizing is not None:
                target_weights = apply_scenario_position_sizing(
                    target_weights,
                    strategy_prices,
                    scenario_sizing,
                    defensive_ticker=strategy.defensive_ticker,
                )
            runtime_results[strategy_name] = run_backtest(
                strategy_name,
                strategy_prices,
                target_weights,
                execution,
                volatility_target=strategy.volatility_target,
                drawdown_control=strategy.drawdown_control,
            )
        return runtime_results

    def _prospective_monitoring_results(
        self,
        baseline_run: Any,
        windows: pd.DataFrame,
    ) -> dict[str, BacktestResult]:
        if windows.empty or "evidence_basis" not in windows:
            return {}
        prospective = windows[
            windows["evidence_basis"].astype(str).eq("prospective_no_backfill")
        ]
        prices = getattr(baseline_run, "prices", pd.DataFrame())
        if prospective.empty or prices.empty:
            return {}
        results: dict[str, BacktestResult] = {}
        cached: dict[str, BacktestResult] = {}
        for _, window in prospective.iterrows():
            strategy = _strategy_from_json(window.get("strategy_json"))
            execution = _execution_from_json(window.get("execution_json"))
            if strategy is None or execution is None:
                continue
            version = str(window.get("strategy_version", ""))
            cache_key = f"{window['strategy_name']}:{version}"
            if cache_key not in cached:
                strategy_prices = _candidate_strategy_prices(prices, strategy)
                if strategy_prices.empty:
                    continue
                target_weights = build_strategy_weights(strategy_prices, strategy)
                cached[cache_key] = run_backtest(
                    str(window["strategy_name"]),
                    strategy_prices,
                    target_weights,
                    execution,
                    volatility_target=strategy.volatility_target,
                    drawdown_control=strategy.drawdown_control,
                )
            results[str(window["window_id"])] = cached[cache_key]
        return results

    def _candidate_manifests_for_strategy_names(self, strategy_names: list[str]) -> pd.DataFrame:
        names = set(strategy_names)
        if not names:
            return pd.DataFrame()

        frames = []
        table = self.read_table("experiment_candidates")
        if not table.empty and "strategy" in table and "strategy_json" in table:
            warehouse_matches = table[table["strategy"].astype(str).isin(names)].copy()
            if not warehouse_matches.empty:
                frames.append(warehouse_matches)

        found_names = (
            set(pd.concat(frames, ignore_index=True)["strategy"].astype(str)) if frames else set()
        )
        missing_names = names - found_names
        if missing_names:
            artifact_matches = _load_candidate_manifests_from_artifacts(missing_names)
            if not artifact_matches.empty:
                frames.append(artifact_matches)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _window_valuations_before(self, window_id: str, valuation_date: str) -> pd.DataFrame:
        valuations = self.read_table("strategy_daily_valuations")
        if valuations.empty:
            return valuations
        return valuations[
            (valuations["window_id"].astype(str) == window_id)
            & (valuations["valuation_date"].astype(str) < valuation_date)
        ].sort_values(["valuation_date", "created_at_utc"])

    def champion_challenger_frame(self) -> pd.DataFrame:
        windows = self.list_monitoring_windows(status="active")
        valuations = self.read_table("strategy_daily_valuations")
        scorecards = self.read_table("experiment_scorecard")
        snapshot_metrics = self.read_table("snapshot_strategy_metrics")
        if windows.empty:
            return pd.DataFrame()

        latest_valuations = pd.DataFrame()
        if not valuations.empty:
            latest_valuations = (
                valuations.sort_values(["window_id", "valuation_date", "created_at_utc"])
                .groupby("window_id", as_index=False)
                .tail(1)
            )

        frame = windows.merge(latest_valuations, on="window_id", how="left", suffixes=("", "_val"))
        if not snapshot_metrics.empty and "strategy" in snapshot_metrics:
            snapshot_scores = snapshot_metrics.rename(
                columns={
                    "cagr": "snapshot_cagr",
                    "sharpe": "snapshot_sharpe",
                    "max_drawdown": "snapshot_max_drawdown",
                    "calmar": "snapshot_calmar",
                    "average_turnover": "snapshot_average_turnover",
                }
            )
            snapshot_columns = [
                "strategy",
                "snapshot_cagr",
                "snapshot_sharpe",
                "snapshot_max_drawdown",
                "snapshot_calmar",
                "snapshot_average_turnover",
            ]
            frame = frame.merge(
                snapshot_scores[
                    [column for column in snapshot_columns if column in snapshot_scores]
                ],
                left_on="strategy_name",
                right_on="strategy",
                how="left",
            ).drop(columns=["strategy"], errors="ignore")
        if not scorecards.empty and "strategy" in scorecards:
            latest_scores = _rank_strategy_rows(scorecards).drop_duplicates(
                "strategy", keep="first"
            )
            score_columns = [
                "strategy",
                "iteration",
                "promotion_decision",
                "promotion_score",
                "selection_adjusted_promotion_score",
                "growth_constrained_utility_score",
                "growth_utility_tier",
                "terminal_wealth_with_contributions_15y",
                "wealth_multiple_vs_spy",
                "wealth_multiple_vs_qqq",
                "drawdown_recovery_return",
                "drawdown_soft_penalty",
                "drawdown_hard_penalty",
                "robustness_score",
                "overfit_risk_score",
                "overfit_risk_label",
                "validation_tier",
                "cagr",
                "sharpe",
                "max_drawdown",
                "calmar",
                "walk_forward_median_cagr",
                "walk_forward_worst_cagr",
                "walk_forward_positive_rate",
                "left_tail_regime_return",
            ]
            frame = frame.merge(
                latest_scores[[column for column in score_columns if column in latest_scores]],
                left_on="strategy_name",
                right_on="strategy",
                how="left",
            )
        frame["forward_status"] = frame.apply(_forward_status, axis=1)
        if "window_role" in frame:
            frame["window_role_sort"] = (
                frame["window_role"].map({"champion": 0, "challenger": 1}).fillna(2)
            )
        sort_columns = [
            column
            for column in [
                "window_role_sort",
                "growth_constrained_utility_score",
                "snapshot_calmar",
                "promotion_score",
                "strategy_name",
            ]
            if column in frame
        ]
        if sort_columns:
            frame = frame.sort_values(
                sort_columns,
                ascending=[True, False, False, False, True][: len(sort_columns)],
                na_position="last",
            )
        return frame

    def list_strategy_registry(self) -> pd.DataFrame:
        return self._query(
            """
            SELECT *
            FROM strategy_registry
            ORDER BY
                CASE status
                    WHEN 'promoted' THEN 0
                    WHEN 'candidate' THEN 1
                    WHEN 'evolve' THEN 2
                    ELSE 3
                END,
                updated_at_utc DESC,
                strategy_name
            """
        )

    def list_monitoring_windows(self, *, status: str | None = "active") -> pd.DataFrame:
        if status is None:
            return self._query(
                """
                SELECT *
                FROM monitoring_windows
                ORDER BY created_at_utc DESC, strategy_name
                """
            )
        return self._query(
            """
            SELECT *
            FROM monitoring_windows
            WHERE status = ?
            ORDER BY created_at_utc DESC, strategy_name
            """,
            [status],
        )

    def read_table(self, table_name: str, *, limit: int | None = None) -> pd.DataFrame:
        if not self.table_exists(table_name):
            return pd.DataFrame()
        limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
        return self._query(f"SELECT * FROM {table_name}{limit_clause}")

    def table_counts(self) -> pd.DataFrame:
        table_names = [
            "run_snapshots",
            "snapshot_jobs",
            "snapshot_strategy_metrics",
            "operating_metric_history",
            "operating_component_history",
            "operating_scenario_driver_history",
            "operating_driver_rotation_history",
            "strategy_registry",
            "monitoring_windows",
            "strategy_daily_valuations",
            "simulation_validation_runs",
            "simulation_validation_metrics",
            "external_macro_videos",
            "external_macro_classifications",
            "external_macro_tradebot_comparisons",
            "experiment_scorecard",
            "experiment_walk_forward_summary",
            "experiment_regime_metrics",
            "journal_decision_snapshots",
            "journal_recommendation_tickets",
            "journal_executions",
        ]
        rows = []
        for table_name in table_names:
            if not self.table_exists(table_name):
                rows.append({"table_name": table_name, "rows": 0})
                continue
            count = int(self._query(f"SELECT COUNT(*) AS rows FROM {table_name}").iloc[0]["rows"])
            rows.append({"table_name": table_name, "rows": count})
        return pd.DataFrame(rows)

    def save_operating_history(
        self,
        *,
        metrics: pd.DataFrame,
        components: pd.DataFrame,
        scenario_drivers: pd.DataFrame,
        driver_rotation: pd.DataFrame,
        replace_sources: Iterable[str] | None = None,
    ) -> dict[str, int]:
        sources = tuple(str(source) for source in (replace_sources or ()))
        if sources:
            placeholders = ", ".join("?" for _source in sources)
            for table_name in (
                "operating_metric_history",
                "operating_component_history",
                "operating_scenario_driver_history",
                "operating_driver_rotation_history",
            ):
                self._execute(
                    f"DELETE FROM {table_name} WHERE source IN ({placeholders})",
                    sources,
                )
        self._upsert_frame("operating_metric_history", metrics, "history_id")
        self._upsert_frame("operating_component_history", components, "history_id")
        self._upsert_frame(
            "operating_scenario_driver_history",
            scenario_drivers,
            "history_id",
        )
        self._upsert_frame(
            "operating_driver_rotation_history",
            driver_rotation,
            "history_id",
        )
        return {
            "operating_metric_history": len(metrics),
            "operating_component_history": len(components),
            "operating_scenario_driver_history": len(scenario_drivers),
            "operating_driver_rotation_history": len(driver_rotation),
        }

    def operating_history_frames(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            self.read_table("operating_metric_history"),
            self.read_table("operating_component_history"),
            self.read_table("operating_scenario_driver_history"),
            self.read_table("operating_driver_rotation_history"),
        )

    def save_external_macro_alignment(
        self,
        *,
        videos: pd.DataFrame,
        classifications: pd.DataFrame,
        comparisons: pd.DataFrame,
    ) -> dict[str, int]:
        classifications = classifications.copy()
        comparisons = comparisons.copy()
        if not classifications.empty:
            if "classification_text_source" not in classifications:
                classifications["classification_text_source"] = "transcript"
            if "classification_confidence" not in classifications:
                classifications["classification_confidence"] = 1.0
        if not comparisons.empty:
            if "classification_text_source" not in comparisons:
                comparisons["classification_text_source"] = "transcript"
            if "classification_confidence" not in comparisons:
                comparisons["classification_confidence"] = 1.0
        self._upsert_frame("external_macro_videos", videos, "video_id")
        self._upsert_frame(
            "external_macro_classifications",
            classifications,
            "classification_id",
        )
        self._upsert_frame(
            "external_macro_tradebot_comparisons",
            comparisons,
            "comparison_id",
        )
        return {
            "external_macro_videos": len(videos),
            "external_macro_classifications": len(classifications),
            "external_macro_tradebot_comparisons": len(comparisons),
        }

    def table_exists(self, table_name: str) -> bool:
        frame = self._query(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = ?
            """,
            [table_name],
        )
        return not frame.empty

    def _ensure_schema(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS warehouse_metadata (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR NOT NULL,
                    updated_at_utc VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO warehouse_metadata
                VALUES ('schema_version', ?, ?)
                """,
                [str(WAREHOUSE_SCHEMA_VERSION), utc_now_iso()],
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_registry (
                    strategy_id VARCHAR PRIMARY KEY,
                    strategy_version VARCHAR NOT NULL,
                    strategy_name VARCHAR NOT NULL,
                    role VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    family VARCHAR NOT NULL,
                    benchmark VARCHAR NOT NULL,
                    universe VARCHAR NOT NULL,
                    params_json VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    updated_at_utc VARCHAR NOT NULL,
                    notes VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_windows (
                    window_id VARCHAR PRIMARY KEY,
                    created_at_utc VARCHAR NOT NULL,
                    updated_at_utc VARCHAR NOT NULL,
                    mode VARCHAR NOT NULL,
                    account VARCHAR NOT NULL,
                    strategy_id VARCHAR NOT NULL,
                    strategy_version VARCHAR NOT NULL,
                    strategy_name VARCHAR NOT NULL,
                    window_role VARCHAR NOT NULL,
                    benchmark VARCHAR NOT NULL,
                    start_date VARCHAR NOT NULL,
                    end_date VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    capital_base DOUBLE NOT NULL,
                    rebalance_cadence VARCHAR NOT NULL,
                    risk_budget VARCHAR NOT NULL,
                    promotion_rule VARCHAR NOT NULL,
                    kill_rule VARCHAR NOT NULL,
                    notes VARCHAR NOT NULL,
                    cohort_id VARCHAR,
                    evidence_basis VARCHAR,
                    historical_backfill_allowed BOOLEAN,
                    strategy_json VARCHAR,
                    execution_json VARCHAR
                )
                """
            )
            self._ensure_table_columns(
                connection,
                "monitoring_windows",
                {
                    "cohort_id": "VARCHAR",
                    "evidence_basis": "VARCHAR",
                    "historical_backfill_allowed": "BOOLEAN",
                    "strategy_json": "VARCHAR",
                    "execution_json": "VARCHAR",
                },
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operating_metric_history (
                    history_id VARCHAR PRIMARY KEY,
                    history_time VARCHAR NOT NULL,
                    snapshot_time VARCHAR NOT NULL,
                    market_date VARCHAR NOT NULL,
                    run_id VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    reconstruction_note VARCHAR NOT NULL,
                    risk_score DOUBLE,
                    one_month_risk_off_probability DOUBLE,
                    risk_budget_multiplier DOUBLE,
                    base_defensive_weight DOUBLE,
                    final_defensive_weight DOUBLE,
                    quantitative_defensive_add_pp DOUBLE,
                    scenario_defensive_add_pp DOUBLE,
                    portfolio_defensive_add_pp DOUBLE,
                    scenario_sizing_authority DOUBLE,
                    scenario_budget_authority DOUBLE,
                    scenario_weighted_stress_authority DOUBLE,
                    portfolio_risk_multiplier DOUBLE,
                    post_expected_shortfall_95 DOUBLE,
                    post_max_stress_loss DOUBLE,
                    post_equity_beta DOUBLE,
                    post_ai_beta DOUBLE,
                    correlation_shift DOUBLE,
                    regime_instability_score DOUBLE,
                    spy_ytd_large_move_share DOUBLE
                )
                """
            )
            self._ensure_table_columns(
                connection,
                "operating_metric_history",
                {
                    "base_defensive_weight": "DOUBLE",
                    "final_defensive_weight": "DOUBLE",
                    "quantitative_defensive_add_pp": "DOUBLE",
                    "scenario_defensive_add_pp": "DOUBLE",
                    "portfolio_defensive_add_pp": "DOUBLE",
                    "scenario_sizing_authority": "DOUBLE",
                    "scenario_budget_authority": "DOUBLE",
                    "scenario_weighted_stress_authority": "DOUBLE",
                },
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operating_component_history (
                    history_id VARCHAR PRIMARY KEY,
                    history_time VARCHAR NOT NULL,
                    snapshot_time VARCHAR NOT NULL,
                    market_date VARCHAR NOT NULL,
                    run_id VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    reconstruction_note VARCHAR NOT NULL,
                    component VARCHAR NOT NULL,
                    component_score DOUBLE,
                    latest_value DOUBLE,
                    state VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operating_scenario_driver_history (
                    history_id VARCHAR PRIMARY KEY,
                    history_time VARCHAR NOT NULL,
                    snapshot_time VARCHAR NOT NULL,
                    market_date VARCHAR NOT NULL,
                    run_id VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    reconstruction_note VARCHAR NOT NULL,
                    driver VARCHAR NOT NULL,
                    score DOUBLE,
                    state VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operating_driver_rotation_history (
                    history_id VARCHAR PRIMARY KEY,
                    history_time VARCHAR NOT NULL,
                    snapshot_time VARCHAR NOT NULL,
                    market_date VARCHAR NOT NULL,
                    run_id VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    reconstruction_note VARCHAR NOT NULL,
                    driver VARCHAR NOT NULL,
                    driver_label VARCHAR NOT NULL,
                    current_activation DOUBLE,
                    proven_relevance DOUBLE,
                    change_30d DOUBLE,
                    change_90d DOUBLE,
                    model_role VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_daily_valuations (
                    valuation_id VARCHAR PRIMARY KEY,
                    window_id VARCHAR NOT NULL,
                    valuation_date VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    strategy_id VARCHAR NOT NULL,
                    strategy_version VARCHAR NOT NULL,
                    strategy_name VARCHAR NOT NULL,
                    mode VARCHAR NOT NULL,
                    account VARCHAR NOT NULL,
                    cohort_id VARCHAR,
                    evidence_basis VARCHAR,
                    historical_backfill_allowed BOOLEAN,
                    equity DOUBLE NOT NULL,
                    cash DOUBLE NOT NULL,
                    gross_exposure DOUBLE NOT NULL,
                    net_exposure DOUBLE NOT NULL,
                    daily_return DOUBLE NOT NULL,
                    cumulative_return DOUBLE NOT NULL,
                    drawdown DOUBLE NOT NULL,
                    benchmark_equity DOUBLE,
                    benchmark_return DOUBLE,
                    benchmark_cumulative_return DOUBLE,
                    excess_return DOUBLE,
                    notes VARCHAR NOT NULL
                )
                """
            )
            self._ensure_table_columns(
                connection,
                "strategy_daily_valuations",
                {
                    "cohort_id": "VARCHAR",
                    "evidence_basis": "VARCHAR",
                    "historical_backfill_allowed": "BOOLEAN",
                    "beta_adjusted_spy_delta": "DOUBLE",
                    "stocks_percent_of_max_sleeve": "DOUBLE",
                    "defensive_percent_of_max_sleeve": "DOUBLE",
                    "gold_percent_of_max_sleeve": "DOUBLE",
                    "crypto_percent_of_max_sleeve": "DOUBLE",
                    "credit_percent_of_max_sleeve": "DOUBLE",
                    "latest_weights_json": "VARCHAR",
                },
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_validation_runs (
                    validation_run_id VARCHAR PRIMARY KEY,
                    created_at_utc VARCHAR NOT NULL,
                    snapshot_run_id VARCHAR NOT NULL,
                    market_date VARCHAR NOT NULL,
                    strategy VARCHAR NOT NULL,
                    reference_strategies VARCHAR NOT NULL,
                    horizons VARCHAR NOT NULL,
                    origin_frequency VARCHAR NOT NULL,
                    min_train_days INTEGER NOT NULL,
                    paths INTEGER NOT NULL,
                    block_days INTEGER NOT NULL,
                    interval_low DOUBLE,
                    interval_high DOUBLE,
                    target_interval_coverage DOUBLE,
                    scenario_history_path VARCHAR NOT NULL,
                    validation_output_path VARCHAR NOT NULL,
                    ablation_output_path VARCHAR NOT NULL,
                    rank_output_path VARCHAR NOT NULL,
                    primary_validity_read VARCHAR NOT NULL,
                    primary_distribution_calibration_read VARCHAR,
                    primary_action_readiness_read VARCHAR,
                    primary_interval_coverage DOUBLE,
                    primary_coverage_error DOUBLE,
                    primary_median_abs_error DOUBLE,
                    primary_launch_decision_accuracy DOUBLE
                )
                """
            )
            self._ensure_table_columns(
                connection,
                "simulation_validation_runs",
                {
                    "interval_low": "DOUBLE",
                    "interval_high": "DOUBLE",
                    "target_interval_coverage": "DOUBLE",
                    "primary_distribution_calibration_read": "VARCHAR",
                    "primary_action_readiness_read": "VARCHAR",
                },
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_validation_metrics (
                    metric_id VARCHAR PRIMARY KEY,
                    validation_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    metric_scope VARCHAR NOT NULL,
                    variant VARCHAR NOT NULL,
                    label VARCHAR NOT NULL,
                    strategy VARCHAR NOT NULL,
                    origin_date VARCHAR NOT NULL,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    train_days INTEGER,
                    paths INTEGER,
                    rows INTEGER,
                    origins INTEGER,
                    horizons_count INTEGER,
                    interval_coverage DOUBLE,
                    target_coverage DOUBLE,
                    coverage_error DOUBLE,
                    median_error_mean DOUBLE,
                    median_abs_error DOUBLE,
                    severe_drawdown_brier DOUBLE,
                    launch_decision_accuracy DOUBLE,
                    launch_action_error_mean DOUBLE,
                    launch_action_score DOUBLE,
                    launch_overrisk_rate DOUBLE,
                    launch_underrisk_rate DOUBLE,
                    bad_action_avoidance_rate DOUBLE,
                    constructive_capture_rate DOUBLE,
                    validity_read VARCHAR NOT NULL,
                    distribution_calibration_read VARCHAR,
                    action_readiness_read VARCHAR,
                    realized_return DOUBLE,
                    realized_max_drawdown DOUBLE,
                    realized_severe_drawdown BOOLEAN,
                    simulated_p10_return DOUBLE,
                    simulated_p50_return DOUBLE,
                    simulated_p90_return DOUBLE,
                    target_interval_coverage DOUBLE,
                    realized_in_interval BOOLEAN,
                    p50_error DOUBLE,
                    p50_abs_error DOUBLE,
                    simulated_severe_drawdown_probability DOUBLE,
                    severe_drawdown_probability_error DOUBLE,
                    simulated_launch_decision VARCHAR NOT NULL,
                    realized_launch_decision VARCHAR NOT NULL,
                    simulated_launch_action INTEGER,
                    realized_launch_action INTEGER,
                    launch_action_error INTEGER,
                    launch_overrisk BOOLEAN,
                    launch_underrisk BOOLEAN,
                    avoided_bad_launch_action BOOLEAN,
                    captured_constructive_launch BOOLEAN,
                    uses_duration_aware_transitions BOOLEAN,
                    uses_covariate_matching BOOLEAN,
                    uses_factor_proxy BOOLEAN
                )
                """
            )
            self._ensure_table_columns(
                connection,
                "simulation_validation_metrics",
                {
                    "launch_action_error_mean": "DOUBLE",
                    "launch_action_score": "DOUBLE",
                    "launch_overrisk_rate": "DOUBLE",
                    "launch_underrisk_rate": "DOUBLE",
                    "bad_action_avoidance_rate": "DOUBLE",
                    "constructive_capture_rate": "DOUBLE",
                    "distribution_calibration_read": "VARCHAR",
                    "action_readiness_read": "VARCHAR",
                    "simulated_launch_action": "INTEGER",
                    "realized_launch_action": "INTEGER",
                    "launch_action_error": "INTEGER",
                    "launch_overrisk": "BOOLEAN",
                    "launch_underrisk": "BOOLEAN",
                    "avoided_bad_launch_action": "BOOLEAN",
                    "captured_constructive_launch": "BOOLEAN",
                },
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_runs (
                    cycle_run_id VARCHAR PRIMARY KEY,
                    created_at_utc VARCHAR NOT NULL,
                    snapshot_run_id VARCHAR NOT NULL,
                    market_date VARCHAR NOT NULL,
                    output_dir VARCHAR NOT NULL,
                    horizons VARCHAR NOT NULL,
                    min_train_days INTEGER NOT NULL,
                    origin_step_days INTEGER NOT NULL,
                    dominant_phase VARCHAR NOT NULL,
                    dominant_phase_probability DOUBLE,
                    candidate_rows INTEGER NOT NULL,
                    frontier_rows INTEGER DEFAULT 0,
                    validation_rows INTEGER NOT NULL,
                    reliability_rows INTEGER DEFAULT 0,
                    crisis_rows INTEGER DEFAULT 0,
                    readout VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_phase_probabilities (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    as_of_date VARCHAR,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    phase VARCHAR NOT NULL,
                    probability DOUBLE,
                    dominant_phase VARCHAR NOT NULL,
                    source VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_transition_forecast (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    phase VARCHAR NOT NULL,
                    probability DOUBLE,
                    dominant_phase VARCHAR NOT NULL,
                    source VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_evidence (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    as_of_date VARCHAR,
                    component VARCHAR NOT NULL,
                    component_score DOUBLE,
                    state VARCHAR NOT NULL,
                    latest_value DOUBLE,
                    interpretation VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_path_state_history (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    as_of_date VARCHAR NOT NULL,
                    evidence_phase VARCHAR NOT NULL,
                    evidence_probability DOUBLE,
                    path_phase VARCHAR NOT NULL,
                    path_probability DOUBLE,
                    previous_path_phase VARCHAR NOT NULL,
                    transition_allowed BOOLEAN,
                    transition_reason VARCHAR,
                    phase_duration_days INTEGER,
                    phase_duration_bucket VARCHAR,
                    prior_unwind_seen_504d BOOLEAN,
                    prior_bottoming_seen_504d BOOLEAN,
                    qqq_drawdown_252d DOUBLE,
                    spy_drawdown_252d DOUBLE,
                    days_since_qqq_peak_252d DOUBLE,
                    normal_cycle_probability DOUBLE,
                    acceleration_probability DOUBLE,
                    pre_break_probability DOUBLE,
                    early_unwind_probability DOUBLE,
                    liquidation_probability DOUBLE,
                    bottoming_probability DOUBLE,
                    recovery_probability DOUBLE,
                    post_unwind_compounding_probability DOUBLE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_path_transition_forecast (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    phase VARCHAR NOT NULL,
                    probability DOUBLE,
                    dominant_phase VARCHAR NOT NULL,
                    current_path_phase VARCHAR NOT NULL,
                    current_phase_duration_days INTEGER,
                    transition_allowed BOOLEAN,
                    precondition VARCHAR,
                    source VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_candidate_scores (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    ticker VARCHAR NOT NULL,
                    asset_role VARCHAR NOT NULL,
                    current_phase VARCHAR NOT NULL,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    candidate_score DOUBLE,
                    candidate_role VARCHAR NOT NULL,
                    current_momentum_21d DOUBLE,
                    current_momentum_63d DOUBLE,
                    current_drawdown_252d DOUBLE,
                    phase_forward_median_return DOUBLE,
                    phase_median_excess_vs_spy DOUBLE,
                    phase_median_excess_vs_qqq DOUBLE,
                    phase_hit_rate_vs_qqq DOUBLE,
                    phase_median_forward_drawdown DOUBLE,
                    phase_origins INTEGER,
                    interpretation VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_phase_candidate_frontier (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    as_of_date VARCHAR,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    phase VARCHAR NOT NULL,
                    phase_probability DOUBLE,
                    ticker VARCHAR NOT NULL,
                    asset_role VARCHAR NOT NULL,
                    exposure_family VARCHAR,
                    frontier_score DOUBLE,
                    validation_score_raw DOUBLE,
                    validation_score DOUBLE,
                    phase_role_fit DOUBLE,
                    origin_confidence DOUBLE,
                    origin_penalty DOUBLE,
                    momentum_score DOUBLE,
                    drawdown_penalty DOUBLE,
                    theme_fragility_penalty DOUBLE,
                    evidence_quality VARCHAR,
                    evidence_flags VARCHAR,
                    phase_window_role VARCHAR,
                    frontier_role VARCHAR NOT NULL,
                    current_momentum_21d DOUBLE,
                    current_momentum_63d DOUBLE,
                    current_drawdown_252d DOUBLE,
                    median_forward_return DOUBLE,
                    median_excess_vs_spy DOUBLE,
                    median_excess_vs_qqq DOUBLE,
                    hit_rate_vs_qqq DOUBLE,
                    median_forward_drawdown DOUBLE,
                    origins INTEGER,
                    interpretation VARCHAR NOT NULL,
                    rank INTEGER
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_validation_metrics (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    dominant_phase VARCHAR NOT NULL,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    ticker VARCHAR NOT NULL,
                    asset_role VARCHAR NOT NULL,
                    origins INTEGER,
                    median_forward_return DOUBLE,
                    mean_forward_return DOUBLE,
                    median_forward_drawdown DOUBLE,
                    worst_forward_drawdown DOUBLE,
                    median_excess_vs_spy DOUBLE,
                    median_excess_vs_qqq DOUBLE,
                    median_excess_vs_bil DOUBLE,
                    hit_rate_vs_spy DOUBLE,
                    hit_rate_vs_qqq DOUBLE,
                    hit_rate_vs_bil DOUBLE,
                    severe_drawdown_rate DOUBLE,
                    phase_rank_score DOUBLE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_path_validation_metrics (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    dominant_phase VARCHAR NOT NULL,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    ticker VARCHAR NOT NULL,
                    asset_role VARCHAR NOT NULL,
                    origins INTEGER,
                    median_forward_return DOUBLE,
                    mean_forward_return DOUBLE,
                    median_forward_drawdown DOUBLE,
                    worst_forward_drawdown DOUBLE,
                    median_excess_vs_spy DOUBLE,
                    median_excess_vs_qqq DOUBLE,
                    median_excess_vs_bil DOUBLE,
                    hit_rate_vs_spy DOUBLE,
                    hit_rate_vs_qqq DOUBLE,
                    hit_rate_vs_bil DOUBLE,
                    severe_drawdown_rate DOUBLE,
                    phase_rank_score DOUBLE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_phase_reliability (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    dominant_phase VARCHAR NOT NULL,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    origins INTEGER,
                    phase_fit_rate DOUBLE,
                    median_phase_probability DOUBLE,
                    median_qqq_forward_return DOUBLE,
                    median_spy_forward_return DOUBLE,
                    median_bil_forward_return DOUBLE,
                    median_qqq_forward_drawdown DOUBLE,
                    severe_qqq_drawdown_rate DOUBLE,
                    expected_behavior VARCHAR NOT NULL,
                    reliability_label VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_path_reliability (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    path_phase VARCHAR NOT NULL,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    origins INTEGER,
                    path_fit_rate DOUBLE,
                    median_path_probability DOUBLE,
                    median_phase_duration_days DOUBLE,
                    median_qqq_forward_return DOUBLE,
                    median_spy_forward_return DOUBLE,
                    median_bil_forward_return DOUBLE,
                    median_qqq_forward_drawdown DOUBLE,
                    expected_behavior VARCHAR NOT NULL,
                    reliability_label VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_tracker_crisis_playback (
                    metric_id VARCHAR PRIMARY KEY,
                    cycle_run_id VARCHAR NOT NULL,
                    created_at_utc VARCHAR NOT NULL,
                    crisis VARCHAR NOT NULL,
                    stage VARCHAR NOT NULL,
                    stage_order INTEGER,
                    origin_date VARCHAR NOT NULL,
                    horizon VARCHAR NOT NULL,
                    horizon_days INTEGER,
                    phase VARCHAR NOT NULL,
                    phase_probability DOUBLE,
                    dominant_phase VARCHAR NOT NULL,
                    dominant_phase_probability DOUBLE,
                    qqq_forward_return DOUBLE,
                    spy_forward_return DOUBLE,
                    bil_forward_return DOUBLE,
                    qqq_forward_drawdown DOUBLE,
                    phase_fit BOOLEAN
                )
                """
            )
            self._ensure_table_columns(
                connection,
                "cycle_tracker_runs",
                {
                    "frontier_rows": "INTEGER DEFAULT 0",
                    "reliability_rows": "INTEGER DEFAULT 0",
                    "crisis_rows": "INTEGER DEFAULT 0",
                },
            )
            self._ensure_table_columns(
                connection,
                "cycle_tracker_phase_candidate_frontier",
                {
                    "phase_window_role": "VARCHAR",
                    "exposure_family": "VARCHAR",
                    "validation_score_raw": "DOUBLE",
                    "validation_score": "DOUBLE",
                    "phase_role_fit": "DOUBLE",
                    "origin_confidence": "DOUBLE",
                    "origin_penalty": "DOUBLE",
                    "momentum_score": "DOUBLE",
                    "drawdown_penalty": "DOUBLE",
                    "theme_fragility_penalty": "DOUBLE",
                    "evidence_quality": "VARCHAR",
                    "evidence_flags": "VARCHAR",
                },
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS external_macro_videos (
                    video_id VARCHAR PRIMARY KEY,
                    source VARCHAR NOT NULL,
                    published_date VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    url VARCHAR NOT NULL,
                    transcript_path VARCHAR NOT NULL,
                    word_count INTEGER,
                    fetched_at_utc VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    error VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS external_macro_classifications (
                    classification_id VARCHAR PRIMARY KEY,
                    video_id VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    published_date VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    macro_posture_score DOUBLE,
                    macro_posture_label VARCHAR NOT NULL,
                    near_term_risk_score DOUBLE,
                    medium_term_bullish_score DOUBLE,
                    large_change_flag BOOLEAN,
                    bullish_term_score DOUBLE,
                    defensive_term_score DOUBLE,
                    key_themes VARCHAR NOT NULL,
                    classification_text_source VARCHAR NOT NULL,
                    classification_confidence DOUBLE,
                    classified_at_utc VARCHAR NOT NULL
                )
                """
            )
            self._ensure_table_columns(
                connection,
                "external_macro_classifications",
                {
                    "classification_text_source": "VARCHAR",
                    "classification_confidence": "DOUBLE",
                },
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS external_macro_tradebot_comparisons (
                    comparison_id VARCHAR PRIMARY KEY,
                    video_id VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    published_date VARCHAR NOT NULL,
                    matched_market_date VARCHAR NOT NULL,
                    matched_source VARCHAR NOT NULL,
                    days_from_tradebot INTEGER,
                    macro_posture_score DOUBLE,
                    macro_posture_label VARCHAR NOT NULL,
                    classification_text_source VARCHAR NOT NULL,
                    classification_confidence DOUBLE,
                    trade_bot_posture_score DOUBLE,
                    trade_bot_posture_label VARCHAR NOT NULL,
                    disagreement DOUBLE,
                    abs_disagreement DOUBLE,
                    disagreement_label VARCHAR NOT NULL,
                    large_change_focus BOOLEAN,
                    trade_bot_risk_score DOUBLE,
                    trade_bot_risk_budget_multiplier DOUBLE,
                    trade_bot_risk_off_probability DOUBLE,
                    trade_bot_portfolio_risk_multiplier DOUBLE,
                    notes VARCHAR NOT NULL,
                    compared_at_utc VARCHAR NOT NULL
                )
                """
            )
            self._ensure_table_columns(
                connection,
                "external_macro_tradebot_comparisons",
                {
                    "classification_text_source": "VARCHAR",
                    "classification_confidence": "DOUBLE",
                },
            )
        finally:
            connection.close()

    @staticmethod
    def _ensure_table_columns(
        connection: duckdb.DuckDBPyConnection,
        table_name: str,
        column_specs: dict[str, str],
    ) -> None:
        existing = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        }
        for column, column_type in column_specs.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {column_type}")

    def _replace_table(self, table_name: str, frame: pd.DataFrame) -> None:
        connection = self._connect()
        try:
            connection.register("replacement_frame", frame)
            connection.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM replacement_frame"
            )
        finally:
            connection.close()

    def _upsert_frame(self, table_name: str, frame: pd.DataFrame, key_column: str) -> None:
        if frame.empty:
            return
        connection = self._connect()
        try:
            table_columns = [
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            ]
            upsert_frame = frame.copy()
            for column in table_columns:
                if column not in upsert_frame:
                    upsert_frame[column] = None
            upsert_frame = upsert_frame[table_columns]
            quoted_columns = ", ".join(_quote_identifier(column) for column in table_columns)
            connection.register("upsert_frame", upsert_frame)
            connection.execute(
                f"""
                DELETE FROM {table_name}
                WHERE {key_column} IN (SELECT {key_column} FROM upsert_frame)
                """
            )
            connection.execute(
                f"INSERT INTO {table_name} ({quoted_columns}) "
                f"SELECT {quoted_columns} FROM upsert_frame"
            )
        finally:
            connection.close()

    def _query(self, query: str, params: Iterable[object] | None = None) -> pd.DataFrame:
        connection = self._connect()
        try:
            return connection.execute(query, list(params or [])).fetchdf()
        finally:
            connection.close()

    def _execute(self, query: str, params: Iterable[object] | None = None) -> None:
        connection = self._connect()
        try:
            connection.execute(query, list(params or []))
        finally:
            connection.close()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=self.read_only)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _strategy_id(strategy_name: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in strategy_name)
    return cleaned.strip("_").lower()[:96]


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _rank_strategy_rows(scorecards: pd.DataFrame) -> pd.DataFrame:
    frame = scorecards.copy()
    if "strategy" not in frame and "name" in frame:
        frame = frame.rename(columns={"name": "strategy"})
    for column in [
        "growth_constrained_utility_score",
        "promotion_score",
        "robustness_score",
        "calmar",
        "iteration",
    ]:
        if column not in frame:
            frame[column] = float("nan")
    return frame.sort_values(
        [
            "growth_constrained_utility_score",
            "promotion_score",
            "robustness_score",
            "calmar",
            "iteration",
        ],
        ascending=False,
        na_position="last",
    )


def _strategy_status(row: pd.Series) -> str:
    decision = str(row.get("promotion_decision", "")).lower()
    if decision == "promote_candidate":
        return "promoted"
    if decision.startswith("evolve"):
        return "evolve"
    if "reject" in decision:
        return "rejected"
    return "candidate"


def _iteration_from_path(path: Path) -> int:
    try:
        return int(path.parent.name.split("_")[-1])
    except (IndexError, ValueError):
        return -1


def _benchmark_equity(baseline_run: Any) -> pd.Series | None:
    prices = baseline_run.prices
    for ticker in ["SPY", "VOO", "VTI", "QQQ"]:
        if ticker in prices:
            return cast(pd.Series, prices[ticker].dropna())
    return None


def _normalize_monitoring_start_date(value: str | None) -> str:
    if value is None or not str(value).strip() or str(value).lower() == "nan":
        return DEFAULT_MONITORING_COHORT_START_DATE
    try:
        return pd.Timestamp(value).date().isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid monitoring start date: {value}") from exc


def _series_with_datetime_index(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    index = pd.to_datetime(clean.index, errors="coerce")
    clean = clean.copy()
    clean.index = index
    clean = clean[~clean.index.isna()].sort_index()
    return clean.astype(float)


def _strategy_path_for_monitoring_window(
    series: pd.Series,
    *,
    start_date: str,
    valuation_date: str,
) -> pd.Series:
    clean = _series_with_datetime_index(series)
    if clean.empty:
        return clean
    start_ts = pd.Timestamp(start_date)
    valuation_ts = pd.Timestamp(valuation_date)
    if valuation_ts < start_ts:
        return clean.iloc[0:0]
    return clean[(clean.index >= start_ts) & (clean.index <= valuation_ts)]


def _series_return_on_or_before(returns: pd.Series, valuation_date: str) -> float:
    clean = _series_with_datetime_index(returns)
    if clean.empty:
        return 0.0
    eligible = clean[clean.index <= pd.Timestamp(valuation_date)]
    if eligible.empty:
        return 0.0
    return float(eligible.iloc[-1])


def _monitoring_window_id(mode: str, account: str, strategy_id: str, start_date: str) -> str:
    account_id = _strategy_id(account) or "account"
    return f"{mode}-{account_id}-{strategy_id}-{start_date}".replace("_", "-")[:120]


def _validate_window_role(role: str) -> None:
    if role not in {"champion", "challenger", "reference"}:
        raise ValueError(f"Unsupported monitoring role: {role}")


def _validate_window_status(status: str) -> None:
    if status not in {"active", "paused", "closed", "killed", "archived"}:
        raise ValueError(f"Unsupported monitoring status: {status}")


def _strategy_from_candidate_manifest(row: pd.Series) -> StrategyConfig | None:
    return _strategy_from_json(row.get("strategy_json"))


def _strategy_from_json(raw: object) -> StrategyConfig | None:
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        return StrategyConfig.model_validate(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _execution_from_json(raw: object) -> ExecutionConfig | None:
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        return ExecutionConfig.model_validate(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _monitoring_valuation_note(window: pd.Series) -> str:
    if str(window.get("evidence_basis", "")) == "prospective_no_backfill":
        return (
            "Prospective frozen-rule valuation from start_date; no earlier history counts as "
            "forward evidence. Actual execution drift is tracked separately in Forward Test."
        )
    return (
        "Start-date anchored reconstructed valuation from the monitoring window start; actual "
        "execution drift is tracked separately in Forward Test."
    )


def _monitoring_backfill_allowed(window: pd.Series) -> bool:
    value = window.get("historical_backfill_allowed", True)
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return True
    return bool(value)


def _scenario_sizing_from_candidate_manifest(row: pd.Series) -> ScenarioSizingConfig | None:
    raw = row.get("scenario_sizing_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(values, dict):
        return None
    try:
        return ScenarioSizingConfig(**values)
    except TypeError:
        return None


def _candidate_strategy_prices(
    prices: pd.DataFrame,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    columns = required_strategy_tickers(strategy)
    if unusable_required_price_columns(prices, columns):
        return pd.DataFrame()
    return prices[columns].dropna(how="all")


def _load_candidate_manifests_from_artifacts(strategy_names: set[str]) -> pd.DataFrame:
    frames = []
    for root in _candidate_manifest_roots():
        if not root.exists():
            continue
        for path in sorted(root.glob("iteration_*/candidates.csv")):
            try:
                frame = pd.read_csv(path)
            except (OSError, pd.errors.ParserError):
                continue
            if "strategy" not in frame:
                continue
            matches = frame[frame["strategy"].astype(str).isin(strategy_names)].copy()
            if matches.empty:
                continue
            if "iteration" not in matches:
                matches.insert(0, "iteration", _iteration_from_path(path))
            if "source_path" not in matches:
                matches.insert(1, "source_path", str(path))
            frames.append(matches)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _candidate_manifest_roots() -> list[Path]:
    roots = []
    for root in (DEFAULT_EXPERIMENTS_DIR, DEFAULT_RESET_EXPERIMENTS_DIR):
        path = Path(root)
        if path not in roots:
            roots.append(path)
    return roots


def _reference_candidate_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    family = frame.get("family", pd.Series("", index=frame.index)).astype(str)
    phase = frame.get("phase", pd.Series("", index=frame.index)).astype(str)
    role = frame.get("role", pd.Series("", index=frame.index)).astype(str)
    strategy_name = frame.get("strategy_name", pd.Series("", index=frame.index)).astype(str)
    return (
        family.eq("reference_portfolio")
        | phase.eq("reference")
        | role.eq("reference_portfolio")
        | strategy_name.str.startswith("i41_ref_")
    )


def _select_monitoring_rows(ranked_rows: pd.DataFrame, limit: int) -> pd.DataFrame:
    if ranked_rows.empty:
        return ranked_rows.copy()
    limit = max(int(limit), 0)
    if limit == 0:
        return ranked_rows.head(0).copy()
    return select_curated_strategy_shelf(ranked_rows, limit=limit)


def _clean_identifier_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).replace({"nan": "", "NaT": "", "None": ""})


def _monitoring_state(row: pd.Series) -> str:
    if bool(row.get("is_valued", False)):
        return "active_valued"
    if bool(row.get("is_active_window", False)) and bool(
        row.get("snapshot_valuation_ready", False)
    ):
        return "active_awaiting_valuation"
    if bool(row.get("is_active_window", False)):
        return "active_research_only"
    if bool(row.get("snapshot_valuation_ready", False)):
        return "available_to_seed_and_value"
    return "available_research_only"


def _forward_drawdown(previous_valuations: pd.DataFrame, current_equity: float) -> float:
    if previous_valuations.empty or "equity" not in previous_valuations:
        return 0.0
    prior_equity = previous_valuations["equity"].astype(float)
    peak = max(float(prior_equity.max()), current_equity)
    if peak <= 0:
        return 0.0
    return current_equity / peak - 1.0


def _latest_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.iloc[-1])


def _latest_gross_exposure(result: Any) -> float:
    weights = getattr(result, "weights", pd.DataFrame())
    if weights.empty:
        return 0.0
    return float(weights.iloc[-1].abs().sum())


def _latest_net_exposure(result: Any) -> float:
    weights = getattr(result, "weights", pd.DataFrame())
    if weights.empty:
        return 0.0
    return float(weights.iloc[-1].sum())


def _latest_strategy_weights(result: Any) -> pd.Series:
    weights = getattr(result, "weights", pd.DataFrame())
    if weights.empty:
        return pd.Series(dtype=float)
    latest = pd.to_numeric(weights.iloc[-1], errors="coerce").fillna(0.0)
    return latest[latest.abs() > 1e-8].astype(float)


def _latest_exposure_diagnostics(prices: pd.DataFrame, weights: pd.Series) -> dict[str, object]:
    if weights.empty:
        return {
            "beta_adjusted_spy_delta": float("nan"),
            "stocks_percent_of_max_sleeve": float("nan"),
            "defensive_percent_of_max_sleeve": float("nan"),
            "gold_percent_of_max_sleeve": float("nan"),
            "crypto_percent_of_max_sleeve": float("nan"),
            "credit_percent_of_max_sleeve": float("nan"),
            "latest_weights_json": "{}",
        }
    sleeve_exposure = build_sleeve_exposure_table(weights, prices)
    diagnostics = {
        "beta_adjusted_spy_delta": aggregate_beta_adjusted_spy_delta(prices, weights),
        "latest_weights_json": json.dumps(
            {str(ticker): float(weight) for ticker, weight in weights.sort_index().items()},
            sort_keys=True,
        ),
    }
    for sleeve in ("stocks", "defensive", "gold", "crypto", "credit"):
        sleeve_row = sleeve_exposure[sleeve_exposure["sleeve"].astype(str) == sleeve]
        diagnostics[f"{sleeve}_percent_of_max_sleeve"] = (
            float(sleeve_row["percent_of_max_sleeve"].iloc[0])
            if not sleeve_row.empty
            else float("nan")
        )
    return diagnostics


def _forward_status(row: pd.Series) -> str:
    if row.get("equity") != row.get("equity"):
        return "awaiting_valuation"
    excess = _optional_float(row.get("excess_return"))
    drawdown = _optional_float(row.get("drawdown"))
    if drawdown is not None and drawdown <= -0.10:
        return "review_drawdown"
    if excess is not None and excess > 0:
        return "ahead_of_benchmark"
    if excess is not None and excess < -0.03:
        return "lagging_benchmark"
    return "in_line"


def _new_validation_run_id() -> str:
    timestamp = utc_now_iso().replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"simval-{timestamp}-{uuid.uuid4().hex[:8]}"


def _new_cycle_run_id() -> str:
    timestamp = utc_now_iso().replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"cycle-{timestamp}-{uuid.uuid4().hex[:8]}"


def _metric_ids(run_id: str, scope: str, count: int) -> list[str]:
    return [f"{run_id}:{scope}:{index:06d}" for index in range(int(count))]


def _simulation_validation_summary_metric_row(
    *,
    validation_run_id: str,
    created_at_utc: str,
    strategy: str,
    metric_scope: str,
    variant: str,
    label: str,
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "metric_id": f"{validation_run_id}:{metric_scope}:{variant}",
        "validation_run_id": validation_run_id,
        "created_at_utc": created_at_utc,
        "metric_scope": metric_scope,
        "variant": variant,
        "label": label,
        "strategy": strategy,
        "origin_date": "",
        "horizon": "all",
        "horizon_days": None,
        "train_days": None,
        "paths": None,
        "rows": _optional_int(summary.get("rows")),
        "origins": _optional_int(summary.get("origins")),
        "horizons_count": _optional_int(summary.get("horizons")),
        "interval_coverage": _optional_float(summary.get("interval_coverage")),
        "target_coverage": _optional_float(summary.get("target_coverage")),
        "coverage_error": _optional_float(summary.get("coverage_error")),
        "median_error_mean": _optional_float(summary.get("median_error_mean")),
        "median_abs_error": _optional_float(summary.get("median_abs_error")),
        "severe_drawdown_brier": _optional_float(summary.get("severe_drawdown_brier")),
        "launch_decision_accuracy": _optional_float(summary.get("launch_decision_accuracy")),
        "launch_action_error_mean": _optional_float(summary.get("launch_action_error_mean")),
        "launch_action_score": _optional_float(summary.get("launch_action_score")),
        "launch_overrisk_rate": _optional_float(summary.get("launch_overrisk_rate")),
        "launch_underrisk_rate": _optional_float(summary.get("launch_underrisk_rate")),
        "bad_action_avoidance_rate": _optional_float(summary.get("bad_action_avoidance_rate")),
        "constructive_capture_rate": _optional_float(summary.get("constructive_capture_rate")),
        "validity_read": str(summary.get("validity_read", "")),
        "distribution_calibration_read": str(
            summary.get("distribution_calibration_read", summary.get("validity_read", ""))
        ),
        "action_readiness_read": str(summary.get("action_readiness_read", "")),
        "realized_return": None,
        "realized_max_drawdown": None,
        "realized_severe_drawdown": None,
        "simulated_p10_return": None,
        "simulated_p50_return": None,
        "simulated_p90_return": None,
        "target_interval_coverage": None,
        "realized_in_interval": None,
        "p50_error": None,
        "p50_abs_error": None,
        "simulated_severe_drawdown_probability": None,
        "severe_drawdown_probability_error": None,
        "simulated_launch_decision": "",
        "realized_launch_decision": "",
        "simulated_launch_action": None,
        "realized_launch_action": None,
        "launch_action_error": None,
        "launch_overrisk": None,
        "launch_underrisk": None,
        "avoided_bad_launch_action": None,
        "captured_constructive_launch": None,
        "uses_duration_aware_transitions": None,
        "uses_covariate_matching": None,
        "uses_factor_proxy": None,
    }


def _simulation_validation_origin_metric_rows(
    *,
    validation_run_id: str,
    created_at_utc: str,
    strategy: str,
    validation: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if validation.empty:
        return rows
    for position, row in validation.reset_index(drop=True).iterrows():
        origin_date = str(row.get("origin_date", ""))
        horizon = str(row.get("horizon", ""))
        rows.append(
            {
                "metric_id": (
                    f"{validation_run_id}:rolling_origin:current_engine:"
                    f"{origin_date}:{horizon}:{position}"
                ),
                "validation_run_id": validation_run_id,
                "created_at_utc": created_at_utc,
                "metric_scope": "rolling_origin",
                "variant": "current_engine",
                "label": "Current simulation engine",
                "strategy": strategy,
                "origin_date": origin_date,
                "horizon": horizon,
                "horizon_days": _optional_int(row.get("horizon_days")),
                "train_days": _optional_int(row.get("train_days")),
                "paths": _optional_int(row.get("paths")),
                "rows": None,
                "origins": None,
                "horizons_count": None,
                "interval_coverage": None,
                "target_coverage": None,
                "coverage_error": None,
                "median_error_mean": None,
                "median_abs_error": None,
                "severe_drawdown_brier": None,
                "launch_decision_accuracy": None,
                "launch_action_error_mean": None,
                "launch_action_score": None,
                "launch_overrisk_rate": None,
                "launch_underrisk_rate": None,
                "bad_action_avoidance_rate": None,
                "constructive_capture_rate": None,
                "validity_read": "",
                "realized_return": _optional_float(row.get("realized_return")),
                "realized_max_drawdown": _optional_float(row.get("realized_max_drawdown")),
                "realized_severe_drawdown": _optional_bool(row.get("realized_severe_drawdown")),
                "simulated_p10_return": _optional_float(row.get("simulated_p10_return")),
                "simulated_p50_return": _optional_float(row.get("simulated_p50_return")),
                "simulated_p90_return": _optional_float(row.get("simulated_p90_return")),
                "target_interval_coverage": _optional_float(row.get("target_interval_coverage")),
                "realized_in_interval": _optional_bool(row.get("realized_in_interval")),
                "p50_error": _optional_float(row.get("p50_error")),
                "p50_abs_error": _optional_float(row.get("p50_abs_error")),
                "simulated_severe_drawdown_probability": _optional_float(
                    row.get("simulated_severe_drawdown_probability")
                ),
                "severe_drawdown_probability_error": _optional_float(
                    row.get("severe_drawdown_probability_error")
                ),
                "simulated_launch_decision": str(row.get("simulated_launch_decision", "")),
                "realized_launch_decision": str(row.get("realized_launch_decision", "")),
                "simulated_launch_action": _optional_int(row.get("simulated_launch_action")),
                "realized_launch_action": _optional_int(row.get("realized_launch_action")),
                "launch_action_error": _optional_int(row.get("launch_action_error")),
                "launch_overrisk": _optional_bool(row.get("launch_overrisk")),
                "launch_underrisk": _optional_bool(row.get("launch_underrisk")),
                "avoided_bad_launch_action": _optional_bool(row.get("avoided_bad_launch_action")),
                "captured_constructive_launch": _optional_bool(
                    row.get("captured_constructive_launch")
                ),
                "uses_duration_aware_transitions": None,
                "uses_covariate_matching": None,
                "uses_factor_proxy": None,
            }
        )
    return rows


def _simulation_horizon_summary_metric_rows(
    *,
    validation_run_id: str,
    created_at_utc: str,
    strategy: str,
    horizon_summary: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in horizon_summary.iterrows():
        horizon = str(row.get("horizon", ""))
        metric = _simulation_validation_summary_metric_row(
            validation_run_id=validation_run_id,
            created_at_utc=created_at_utc,
            strategy=strategy,
            metric_scope="horizon_summary",
            variant="current_engine",
            label=f"{horizon} horizon",
            summary=row.to_dict(),
        )
        metric["metric_id"] = f"{validation_run_id}:horizon_summary:current_engine:{horizon}"
        metric["horizon"] = horizon
        metric["horizon_days"] = _optional_int(row.get("horizon_days"))
        rows.append(metric)
    return rows


def _simulation_ablation_metric_rows(
    *,
    validation_run_id: str,
    created_at_utc: str,
    strategy: str,
    ablation_summary: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in ablation_summary.iterrows():
        variant = str(row.get("variant", ""))
        rows.append(
            {
                **_simulation_validation_summary_metric_row(
                    validation_run_id=validation_run_id,
                    created_at_utc=created_at_utc,
                    strategy=strategy,
                    metric_scope="ablation_summary",
                    variant=variant,
                    label=str(row.get("label", variant)),
                    summary=row.to_dict(),
                ),
                "uses_duration_aware_transitions": _optional_bool(
                    row.get("uses_duration_aware_transitions")
                ),
                "uses_covariate_matching": _optional_bool(row.get("uses_covariate_matching")),
                "uses_factor_proxy": _optional_bool(row.get("uses_factor_proxy")),
            }
        )
    return rows


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return bool(numeric)


def _optional_int(value: object) -> int | None:
    numeric = _optional_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric
