from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.config import ExecutionConfig, StrategyConfig
from trade_bot.DEFAULTS import (
    DEFAULT_EXPERIMENT_REGISTRY_LIMIT,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MONITORING_TOP_N,
    DEFAULT_RESET_EXPERIMENTS_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
)
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

    def __init__(self, db_path: str | Path = DEFAULT_RUN_STORE_DB_PATH) -> None:
        self.db_path = Path(db_path)
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
                "Snapshot-operable strategy" f"; CAGR={cagr:.2%}"
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
        today = start_date or date.today().isoformat()
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

        existing = self.list_monitoring_windows(status=None)
        if not existing.empty:
            existing_active = existing[
                (existing["mode"].astype(str) == mode)
                & (existing["account"].astype(str) == account)
                & (existing["strategy_id"].astype(str) == strategy_id)
                & (existing["status"].astype(str) == "active")
            ]
            if not existing_active.empty:
                window = existing_active.sort_values(["start_date", "created_at_utc"]).iloc[-1]
                self.update_monitoring_window(
                    str(window["window_id"]),
                    role=role,
                    status="active",
                    capital_base=capital_base,
                    demote_other_champions=demote_other_champions,
                )
                return MonitoringWindowSeedResult(
                    window_id=str(window["window_id"]),
                    strategy_id=strategy_id,
                    strategy_name=str(row["strategy_name"]),
                    role=role,
                )

        today = start_date or date.today().isoformat()
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

        params.append(window_id)
        self._execute(
            f"UPDATE monitoring_windows SET {', '.join(updates)} WHERE window_id = ?",
            params,
        )
        if role == "champion" and demote_other_champions:
            self._demote_other_champions(
                window_id,
                str(row["mode"]),
                str(row["account"]),
            )
        return True

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
        merged["monitoring_sort_score"] = (
            merged["growth_constrained_utility_score"]
            .fillna(merged["snapshot_calmar"])
            .fillna(merged["selection_adjusted_promotion_score"])
            .fillna(merged["promotion_score"])
        )
        return merged.sort_values(
            [
                "status_rank",
                "validation_rank",
                "monitoring_sort_score",
                "selection_adjusted_promotion_score",
                "promotion_score",
                "strategy_name",
            ],
            ascending=[True, True, False, False, False, True],
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
        rows = []
        for _, window in windows.iterrows():
            strategy_name = str(window["strategy_name"])
            if strategy_name not in runtime_results:
                continue
            result = runtime_results[strategy_name]
            if result.equity.empty:
                continue
            equity_series = result.equity.dropna()
            returns = result.returns.dropna()
            if equity_series.empty:
                continue
            capital_base = float(window["capital_base"])
            previous_valuations = self._window_valuations_before(
                str(window["window_id"]),
                valuation_date,
            )
            has_previous_valuation = not previous_valuations.empty
            latest_strategy_return = float(returns.iloc[-1]) if not returns.empty else 0.0
            benchmark_return = 0.0
            if benchmark_equity is not None and not benchmark_equity.empty:
                aligned_benchmark = benchmark_equity.reindex(equity_series.index).ffill().dropna()
                if len(aligned_benchmark) >= 2:
                    benchmark_return = float(aligned_benchmark.pct_change().iloc[-1])

            if has_previous_valuation:
                previous_row = previous_valuations.iloc[-1]
                daily_return = latest_strategy_return
                paper_equity = float(previous_row["equity"]) * (1.0 + daily_return)
                previous_benchmark_equity = _optional_float(previous_row.get("benchmark_equity"))
                benchmark_base = previous_benchmark_equity or capital_base
                benchmark_equity_value = benchmark_base * (1.0 + benchmark_return)
            else:
                daily_return = 0.0
                benchmark_return = 0.0
                paper_equity = capital_base
                benchmark_equity_value = capital_base

            cumulative_return = paper_equity / capital_base - 1.0
            benchmark_cumulative_return = benchmark_equity_value / capital_base - 1.0
            drawdown = _forward_drawdown(previous_valuations, paper_equity)
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
                    "notes": "Forward paper valuation compounded from monitoring-window start; first valuation starts at capital base.",
                }
            )
        if not rows:
            return 0
        self._upsert_frame("strategy_daily_valuations", pd.DataFrame(rows), "valuation_id")
        return len(rows)

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
                strategy.tickers,
                strategy.defensive_ticker,
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
            "strategy_registry",
            "monitoring_windows",
            "strategy_daily_valuations",
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
                    notes VARCHAR NOT NULL
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
                    "beta_adjusted_spy_delta": "DOUBLE",
                    "stocks_percent_of_max_sleeve": "DOUBLE",
                    "defensive_percent_of_max_sleeve": "DOUBLE",
                    "gold_percent_of_max_sleeve": "DOUBLE",
                    "crypto_percent_of_max_sleeve": "DOUBLE",
                    "credit_percent_of_max_sleeve": "DOUBLE",
                    "latest_weights_json": "VARCHAR",
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
        return duckdb.connect(str(self.db_path))


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
    raw = row.get("strategy_json")
    if not isinstance(raw, str) or not raw or raw == "nan":
        return None
    try:
        return StrategyConfig.model_validate(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


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
    tickers: list[str],
    defensive_ticker: str | None,
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*tickers, *([defensive_ticker] if defensive_ticker else [])]))
    available_columns = [column for column in columns if column in prices.columns]
    if len(available_columns) != len(columns):
        return pd.DataFrame()
    return prices[available_columns].dropna(how="all")


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
    experiment_rows = ranked_rows[ranked_rows["source"].astype(str) == "experiment_scorecard"]
    if len(experiment_rows) >= limit:
        return select_curated_strategy_shelf(experiment_rows, limit=limit)
    remainder = ranked_rows.drop(index=experiment_rows.index)
    combined = pd.concat([experiment_rows, remainder], ignore_index=True)
    return select_curated_strategy_shelf(combined, limit=limit)


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


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric
