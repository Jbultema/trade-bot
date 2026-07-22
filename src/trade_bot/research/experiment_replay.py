from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trade_bot.config import BotConfig, configured_tickers
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.research.approach_explorer import (
    decision_sanity_from_catalog_row,
    future_state_model_from_catalog_row,
    scenario_sizing_from_catalog_row,
    strategy_drawdown_model_from_catalog_row,
    strategy_from_catalog_row,
)
from trade_bot.research.artifact_provenance import (
    research_config_sha256,
    research_source_tree_sha256,
    verify_research_manifest,
)
from trade_bot.research.evaluation_contract import (
    EVALUATION_CONTRACT_SCHEMA_VERSION,
    build_strategy_evaluation_contract,
    evaluation_contract_sha256,
)
from trade_bot.research.experiments import (
    ExperimentCandidate,
    _candidate_tickers,
    evaluate_experiment_candidates,
)

EXPERIMENT_LIBRARY_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentLibraryReplay:
    source_root: Path
    output_root: Path
    iterations: tuple[int, ...]
    candidate_count: int
    evaluation_contract_sha256: str
    source_tree_sha256: str


def replay_experiment_library(
    config: BotConfig,
    *,
    source_root: str | Path,
    output_root: str | Path,
    refresh_data: bool = False,
    max_workers: int = 4,
    progress: Callable[[int, int, int], None] | None = None,
) -> ExperimentLibraryReplay:
    """Replay every saved candidate definition under one evaluation contract."""

    source = Path(source_root)
    output = Path(output_root)
    if not source.is_dir():
        raise FileNotFoundError(f"Experiment source directory does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Replay output must be absent or empty so a partial library cannot be mixed: {output}"
        )

    candidates_by_iteration = load_saved_experiment_candidates(source)
    if not candidates_by_iteration:
        raise ValueError(f"No saved candidates found under {source}")
    saved_candidate_names = [
        candidate.name
        for candidates in candidates_by_iteration.values()
        for candidate in candidates
    ]
    configured_candidates = _configured_candidates(config, candidates_by_iteration)
    if configured_candidates:
        candidates_by_iteration = {0: configured_candidates, **candidates_by_iteration}
    candidate_names = [
        candidate.name
        for candidates in candidates_by_iteration.values()
        for candidate in candidates
    ]
    if len(candidate_names) != len(set(candidate_names)):
        duplicates = sorted(
            name for name in set(candidate_names) if candidate_names.count(name) > 1
        )
        raise ValueError(f"Saved candidate names are not unique: {duplicates[:10]}")

    all_candidates = tuple(
        candidate
        for candidates in candidates_by_iteration.values()
        for candidate in candidates
    )
    tickers = sorted(set(configured_tickers(config)) | _candidate_tickers(all_candidates))
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    )
    contract = build_strategy_evaluation_contract(config, prices)
    contract_hash = evaluation_contract_sha256(contract)
    output.mkdir(parents=True, exist_ok=True)

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    def evaluate_iteration(
        iteration: int,
        candidates: tuple[ExperimentCandidate, ...],
    ) -> int:
        batch = evaluate_experiment_candidates(
            config,
            iteration=iteration,
            candidates=candidates,
            prices=prices,
            output_dir=output,
        )
        hashes = set(batch.scorecard["evaluation_contract_sha256"].astype(str))
        if hashes != {contract_hash}:
            raise RuntimeError(
                f"Iteration {iteration} produced a mismatched evaluation contract: {hashes}"
            )
        return iteration

    completed_iterations: list[int] = []
    items = list(candidates_by_iteration.items())
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        futures = {
            executor.submit(evaluate_iteration, iteration, candidates): iteration
            for iteration, candidates in items
        }
        for future in as_completed(futures):
            completed_iterations.append(future.result())
            if progress is not None:
                progress(
                    len(completed_iterations),
                    len(items),
                    futures[future],
                )
    completed_iterations.sort()

    manifest = {
        "schema_version": EXPERIMENT_LIBRARY_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_root": str(source),
        "output_root": str(output),
        "replay_semantics": "exact_saved_candidate_definitions",
        "expected_candidate_count": len(candidate_names),
        "saved_candidate_count": len(saved_candidate_names),
        "configured_candidate_count": len(configured_candidates),
        "expected_iterations": completed_iterations,
        "evaluation_contract_schema_version": EVALUATION_CONTRACT_SCHEMA_VERSION,
        "evaluation_contract_sha256": contract_hash,
        "evaluation_contract": contract,
        "source_tree_sha256": research_source_tree_sha256(),
        "config_sha256": research_config_sha256(config),
        "candidate_roster_sha256": _json_sha256(candidate_names),
        "iteration_manifests": {
            str(iteration): _file_sha256(output / f"iteration_{iteration:02d}" / "manifest.json")
            for iteration in completed_iterations
        },
    }
    (output / "library_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verification = verify_experiment_library(output, config=config)
    if verification["status"] != "current_complete":
        raise RuntimeError(f"Replayed experiment library failed verification: {verification}")

    return ExperimentLibraryReplay(
        source_root=source,
        output_root=output,
        iterations=tuple(completed_iterations),
        candidate_count=len(candidate_names),
        evaluation_contract_sha256=contract_hash,
        source_tree_sha256=str(manifest["source_tree_sha256"]),
    )


def load_saved_experiment_candidates(
    root: str | Path,
) -> dict[int, tuple[ExperimentCandidate, ...]]:
    source = Path(root)
    output: dict[int, tuple[ExperimentCandidate, ...]] = {}
    for candidate_path in sorted(source.glob("iteration_*/candidates.csv")):
        iteration = _iteration_from_path(candidate_path)
        if iteration < 0:
            continue
        frame = pd.read_csv(candidate_path, dtype=str, keep_default_na=False)
        candidates = tuple(_candidate_from_row(row) for _, row in frame.iterrows())
        if candidates:
            output[iteration] = candidates
    return dict(sorted(output.items()))


def verify_experiment_library(
    root: str | Path,
    *,
    config: BotConfig | None = None,
) -> dict[str, object]:
    library_root = Path(root)
    manifest_path = library_root / "library_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing_or_unreadable_manifest"}
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        return {"status": "incomplete_manifest"}

    current_source = research_source_tree_sha256()
    if manifest.get("source_tree_sha256") != current_source:
        return {
            "status": "stale_source_tree",
            "recorded_source_tree_sha256": manifest.get("source_tree_sha256"),
            "current_source_tree_sha256": current_source,
        }
    if config is not None and manifest.get("config_sha256") != research_config_sha256(config):
        return {"status": "stale_configuration"}
    expected_iterations = manifest.get("expected_iterations")
    if not isinstance(expected_iterations, list) or not expected_iterations:
        return {"status": "missing_iteration_roster"}
    expected_contract = str(manifest.get("evaluation_contract_sha256", ""))
    expected_manifest_hashes = manifest.get("iteration_manifests", {})
    if not isinstance(expected_manifest_hashes, dict):
        return {"status": "missing_iteration_manifest_hashes"}
    observed_rows = 0
    for raw_iteration in expected_iterations:
        iteration = int(raw_iteration)
        iteration_root = library_root / f"iteration_{iteration:02d}"
        iteration_manifest_path = iteration_root / "manifest.json"
        if not iteration_manifest_path.is_file():
            return {"status": "missing_iteration_manifest", "iteration": iteration}
        if expected_manifest_hashes.get(str(iteration)) != _file_sha256(
            iteration_manifest_path
        ):
            return {"status": "iteration_manifest_hash_mismatch", "iteration": iteration}
        verification = verify_research_manifest(
            iteration_manifest_path,
            current_source_tree_sha256=current_source,
        )
        if verification.get("artifact_integrity_status") != "verified":
            return {
                "status": "iteration_artifact_failure",
                "iteration": iteration,
                "verification": verification,
            }
        if verification.get("source_tree_status") != "current":
            return {"status": "iteration_source_mismatch", "iteration": iteration}
        scorecard_path = iteration_root / "scorecard.csv"
        frame = pd.read_csv(scorecard_path)
        contracts = set(frame.get("evaluation_contract_sha256", pd.Series(dtype=str)).astype(str))
        if contracts != {expected_contract}:
            return {
                "status": "iteration_contract_mismatch",
                "iteration": iteration,
                "contracts": sorted(contracts),
            }
        observed_rows += len(frame)
    expected_rows = int(manifest.get("expected_candidate_count", -1))
    if observed_rows != expected_rows:
        return {
            "status": "candidate_count_mismatch",
            "expected_candidate_count": expected_rows,
            "observed_candidate_count": observed_rows,
        }
    return {
        "status": "current_complete",
        "candidate_count": observed_rows,
        "iteration_count": len(expected_iterations),
        "evaluation_contract_sha256": expected_contract,
        "source_tree_sha256": current_source,
    }


def _candidate_from_row(row: pd.Series) -> ExperimentCandidate:
    parent = str(row.get("parent", "")).strip()
    return ExperimentCandidate(
        name=str(row["strategy"]),
        hypothesis=str(row.get("hypothesis", "")),
        role=str(row.get("role", "unknown")),
        strategy=strategy_from_catalog_row(row),
        scenario_sizing=scenario_sizing_from_catalog_row(row),
        future_state_model=future_state_model_from_catalog_row(row),
        strategy_drawdown_model=strategy_drawdown_model_from_catalog_row(row),
        decision_sanity=decision_sanity_from_catalog_row(row),
        phase=str(row.get("phase", "legacy")),
        family=str(row.get("family", "unknown")),
        parent=parent or None,
    )


def _configured_candidates(
    config: BotConfig,
    saved: dict[int, tuple[ExperimentCandidate, ...]],
) -> tuple[ExperimentCandidate, ...]:
    saved_by_name = {
        candidate.name: candidate
        for candidates in saved.values()
        for candidate in candidates
    }
    candidates: list[ExperimentCandidate] = []
    for name, strategy in config.strategies.items():
        if name in saved_by_name:
            if saved_by_name[name].strategy != strategy:
                raise ValueError(
                    f"Configured strategy {name!r} collides with a different saved definition."
                )
            continue
        candidates.append(
            ExperimentCandidate(
                name=name,
                hypothesis="Current configured strategy added to the canonical comparison library.",
                role="configured_strategy",
                strategy=strategy,
                phase="configured",
                family="baseline_runtime",
            )
        )
    return tuple(candidates)


def _iteration_from_path(path: Path) -> int:
    try:
        return int(path.parent.name.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return -1


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
