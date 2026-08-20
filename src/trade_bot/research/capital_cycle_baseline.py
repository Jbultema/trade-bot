from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
)
from trade_bot.storage.run_store import RunStore

DEFAULT_FROZEN_BASELINE_DIR = Path("reports/capital_cycle_baseline")


@dataclass(frozen=True)
class FrozenTradeBotBaseline:
    baseline: pd.DataFrame
    baseline_path: Path
    manifest_path: Path


def freeze_trade_bot_snapshot_baseline(
    *,
    store: str | Path = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: str | Path = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: str | Path = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    output_dir: str | Path = DEFAULT_FROZEN_BASELINE_DIR,
    version: str = "trade_bot_snapshot_baseline_2026_08_19",
) -> FrozenTradeBotBaseline:
    """Freeze saved Trade-bot outputs as a comparator; never refit them in folds."""

    run_store = RunStore(
        store,
        artifact_dir=artifact_dir,
        job_log_dir=job_log_dir,
        read_only=True,
    )
    snapshots = run_store.list_snapshots(limit=100_000).copy()
    if snapshots.empty:
        raise ValueError("No saved Trade-bot snapshots are available to freeze.")
    snapshots["market_date"] = pd.to_datetime(snapshots["market_date"], errors="coerce")
    snapshots["stored_at_utc"] = pd.to_datetime(
        snapshots.get("stored_at_utc", snapshots["created_at_utc"]),
        errors="coerce",
        utc=True,
    )
    snapshots = snapshots.dropna(subset=["market_date"]).sort_values(
        ["market_date", "stored_at_utc", "created_at_utc", "run_id"]
    )
    snapshots = snapshots.drop_duplicates("market_date", keep="last")
    columns = [
        "market_date",
        "run_id",
        "created_at_utc",
        "stored_at_utc",
        "risk_status",
        "recommended_action",
        "risk_budget_multiplier",
        "combined_config_hash",
        "schema_version",
        "provenance_json",
    ]
    baseline = snapshots[[column for column in columns if column in snapshots]].copy()
    baseline["risk_budget_multiplier"] = pd.to_numeric(
        baseline["risk_budget_multiplier"], errors="coerce"
    )
    baseline["risk_reduction"] = 1.0 - baseline["risk_budget_multiplier"]
    baseline["baseline_version"] = version
    baseline["baseline_semantics"] = (
        "saved_or_reconstructed_snapshot_output_not_contemporaneous_model-version_history"
    )
    baseline["allocation_authority"] = 0.0
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    baseline_path = destination / "frozen_trade_bot_baseline.csv"
    baseline.to_csv(baseline_path, index=False)
    digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    manifest = {
        "version": version,
        "rows": len(baseline),
        "first_market_date": str(baseline["market_date"].min().date()),
        "last_market_date": str(baseline["market_date"].max().date()),
        "sha256": digest,
        "selection": "latest stored snapshot per market_date",
        "fold_policy": "raw baseline values are frozen and may not be refit inside validation folds",
        "interpretation_limit": (
            "Historical as-of snapshots may have been reconstructed using the current engine; "
            "they are not evidence of what the live model actually said before Trade-bot existed."
        ),
        "allocation_authority": 0.0,
    }
    manifest_path = destination / "frozen_trade_bot_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return FrozenTradeBotBaseline(
        baseline=baseline,
        baseline_path=baseline_path,
        manifest_path=manifest_path,
    )
