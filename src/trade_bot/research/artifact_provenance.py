from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from trade_bot.research.research_governance import audit_point_in_time_universe

RESEARCH_MANIFEST_SCHEMA_VERSION = 3


def build_runtime_provenance(
    prices: pd.DataFrame | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Return inspectable code, dependency, and input identity for persisted state."""

    root = Path(repo_root) if repo_root is not None else _repo_root()
    return {
        **_git_metadata(root),
        "source_tree_sha256": _source_tree_sha256(root),
        "poetry_lock_sha256": _optional_file_sha256(root / "poetry.lock"),
        "pyproject_sha256": _optional_file_sha256(root / "pyproject.toml"),
        "price_input": _price_metadata(prices),
    }


def write_research_manifest(
    output_dir: str | Path,
    *,
    study: str,
    config: BaseModel | Mapping[str, object],
    prices: pd.DataFrame | None = None,
    universe_membership: pd.DataFrame | None = None,
    parameters: Mapping[str, object] | None = None,
    artifacts: Sequence[str] = (),
) -> Path:
    """Write a reproducibility manifest beside a persisted research result."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_payload = _jsonable_config(config)
    repo_root = _repo_root()
    runtime_provenance = build_runtime_provenance(prices, repo_root=repo_root)
    artifact_names = sorted(dict.fromkeys(str(item) for item in artifacts))
    parameter_payload = _jsonable(parameters or {})
    universe_audit = audit_point_in_time_universe(prices, universe_membership)
    trial_roster, trial_roster_source = _manifest_trial_roster(study, parameter_payload)
    manifest = {
        "schema_version": RESEARCH_MANIFEST_SCHEMA_VERSION,
        "study": study,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "research_status": "retrospective_research_only",
        "automatic_promotion_allowed": False,
        "config_sha256": _json_sha256(config_payload),
        "config": config_payload,
        "parameters": parameter_payload,
        "price_input": _price_metadata(prices),
        "code": {
            key: value
            for key, value in runtime_provenance.items()
            if key != "price_input"
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "trade_bot_version": _package_version(),
            "pandas_version": pd.__version__,
        },
        "artifacts": artifact_names,
        "artifact_integrity": _artifact_integrity(output, artifact_names),
        "research_governance": {
            "point_in_time_universe": universe_audit,
            "trial_roster": {
                "status": (
                    "explicit_roster_recorded"
                    if trial_roster_source != "study_level_fallback"
                    else "study_level_only"
                ),
                "source": trial_roster_source,
                "declared_trial_count": len(trial_roster),
                "roster_sha256": _json_sha256(trial_roster),
            },
            "promotion_evidence_gate": (
                "eligible_for_human_review"
                if bool(universe_audit.get("promotion_eligible"))
                and trial_roster_source != "study_level_fallback"
                else "blocked_incomplete_universe_or_trial_history"
            ),
        },
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_research_manifest(
    manifest_path: str | Path,
    *,
    current_source_tree_sha256: str | None = None,
) -> dict[str, object]:
    """Verify persisted artifact bytes and compare the recorded source tree to current code."""

    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "artifact_integrity_status": "manifest_unreadable",
            "declared_artifact_count": 0,
            "verified_artifact_count": 0,
            "missing_artifact_count": 0,
            "artifact_mismatch_count": 0,
            "unverified_artifact_count": 0,
            "source_tree_status": "unavailable",
        }
    if not isinstance(manifest, dict):
        return {
            "artifact_integrity_status": "manifest_unreadable",
            "declared_artifact_count": 0,
            "verified_artifact_count": 0,
            "missing_artifact_count": 0,
            "artifact_mismatch_count": 0,
            "unverified_artifact_count": 0,
            "source_tree_status": "unavailable",
        }

    artifact_names = _manifest_artifact_names(manifest)
    integrity_rows = manifest.get("artifact_integrity", [])
    integrity_by_path = (
        {
            str(row.get("path", "")): row
            for row in integrity_rows
            if isinstance(row, dict) and str(row.get("path", ""))
        }
        if isinstance(integrity_rows, list)
        else {}
    )
    verified = 0
    missing = 0
    mismatched = 0
    unverified = 0
    for artifact_name in artifact_names:
        artifact_path = _declared_artifact_path(path.parent, artifact_name)
        expected = integrity_by_path.get(artifact_name)
        if artifact_path is None or not artifact_path.is_file():
            missing += 1
            continue
        if not isinstance(expected, dict):
            unverified += 1
            continue
        expected_size = expected.get("size_bytes")
        expected_sha256 = expected.get("sha256")
        if not isinstance(expected_size, int) or not isinstance(expected_sha256, str):
            unverified += 1
            continue
        if (
            artifact_path.stat().st_size != expected_size
            or _file_sha256(artifact_path) != expected_sha256
        ):
            mismatched += 1
            continue
        verified += 1

    if missing:
        integrity_status = "missing_artifacts"
    elif mismatched:
        integrity_status = "hash_or_size_mismatch"
    elif unverified or not artifact_names:
        integrity_status = "unverified_no_hashes"
    else:
        integrity_status = "verified"

    code = manifest.get("code", {})
    recorded_source_hash = code.get("source_tree_sha256") if isinstance(code, dict) else None
    current_source_hash = (
        current_source_tree_sha256
        if current_source_tree_sha256 is not None
        else research_source_tree_sha256()
    )
    if (
        not isinstance(recorded_source_hash, str)
        or not recorded_source_hash
        or not current_source_hash
    ):
        source_tree_status = "unavailable"
    elif recorded_source_hash == current_source_hash:
        source_tree_status = "current"
    else:
        source_tree_status = "stale"
    return {
        "artifact_integrity_status": integrity_status,
        "declared_artifact_count": len(artifact_names),
        "verified_artifact_count": verified,
        "missing_artifact_count": missing,
        "artifact_mismatch_count": mismatched,
        "unverified_artifact_count": unverified,
        "source_tree_status": source_tree_status,
    }


def research_source_tree_sha256() -> str:
    """Return the source/config identity used by newly written research manifests."""

    return _source_tree_sha256(_repo_root())


def _artifact_integrity(output: Path, artifact_names: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact_name in artifact_names:
        artifact_path = _declared_artifact_path(output, artifact_name)
        if artifact_path is None or not artifact_path.is_file():
            rows.append(
                {
                    "path": artifact_name,
                    "size_bytes": None,
                    "sha256": None,
                }
            )
            continue
        rows.append(
            {
                "path": artifact_name,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": _file_sha256(artifact_path),
            }
        )
    return rows


def _manifest_artifact_names(manifest: Mapping[str, object]) -> list[str]:
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        return []
    return sorted(dict.fromkeys(str(item) for item in artifacts if str(item)))


def _declared_artifact_path(output: Path, artifact_name: str) -> Path | None:
    relative = Path(artifact_name)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return output / relative


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_sha256(path: Path) -> str | None:
    return _file_sha256(path) if path.is_file() else None


def _jsonable_config(config: BaseModel | Mapping[str, object]) -> dict[str, object]:
    if isinstance(config, BaseModel):
        return _jsonable(config.model_dump(mode="json"))
    return _jsonable(dict(config))


def _manifest_trial_roster(
    study: str,
    parameters: Mapping[str, Any],
) -> tuple[list[str], str]:
    for key in ("candidate_names", "candidate_set", "mechanisms", "strategies"):
        values = parameters.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            candidates = [str(value) for value in values if str(value)]
            if candidates:
                return list(dict.fromkeys(candidates)), key
    strategy = parameters.get("strategy")
    if strategy:
        return [str(strategy)], "strategy"
    return [study], "study_level_fallback"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _price_metadata(prices: pd.DataFrame | None) -> dict[str, object]:
    if prices is None:
        return {}
    ordered = prices.sort_index().sort_index(axis=1)
    value_hash = pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
    digest = hashlib.sha256()
    digest.update(json.dumps([str(column) for column in ordered.columns]).encode())
    digest.update(value_hash)
    valid_dates = pd.to_datetime(ordered.index, errors="coerce")
    valid_dates = valid_dates[~pd.isna(valid_dates)]
    return {
        "rows": int(len(ordered)),
        "columns": [str(column) for column in ordered.columns],
        "start_date": valid_dates.min().date().isoformat() if len(valid_dates) else None,
        "market_date": valid_dates.max().date().isoformat() if len(valid_dates) else None,
        "frame_sha256": digest.hexdigest(),
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_metadata(repo_root: Path) -> dict[str, object]:
    sha = _git_output(repo_root, "rev-parse", "HEAD")
    tree_sha = _git_output(repo_root, "rev-parse", "HEAD^{tree}")
    status = _git_output(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_sha": sha or None,
        "git_tree_sha": tree_sha or None,
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _source_tree_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    roots = [repo_root / "src" / "trade_bot", repo_root / "configs"]
    paths = [repo_root / "pyproject.toml", repo_root / "poetry.lock"]
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(paths):
        if not path.exists():
            continue
        relative = path.relative_to(repo_root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _package_version() -> str:
    try:
        return version("trade-bot")
    except PackageNotFoundError:
        return "unknown"
