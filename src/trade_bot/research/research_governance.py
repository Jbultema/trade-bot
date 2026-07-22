from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

POINT_IN_TIME_UNIVERSE_COLUMNS = (
    "ticker",
    "effective_from",
    "effective_to",
    "source",
    "source_as_of",
    "delisting_return_included",
    "delisting_return_source",
)
RETROSPECTIVE_ADAPTATION_LIMIT = 25


def audit_point_in_time_universe(
    prices: pd.DataFrame | None,
    membership: pd.DataFrame | None = None,
    *,
    required_tickers: Sequence[str] | None = None,
    weights: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Fail-closed audit for historical membership and delisting evidence."""

    tickers = sorted(
        dict.fromkeys(
            str(ticker)
            for ticker in (
                required_tickers
                if required_tickers is not None
                else (prices.columns if isinstance(prices, pd.DataFrame) else ())
            )
        )
    )
    if membership is None or membership.empty:
        return {
            "status": "missing_point_in_time_membership",
            "promotion_eligible": False,
            "required_ticker_count": len(tickers),
            "covered_ticker_count": 0,
            "missing_tickers": tickers,
            "holding_membership_violation_count": None,
            "delisting_treatment_status": "unverified",
            "delisting_evidence_gap_count": None,
        }

    missing_columns = [
        column for column in POINT_IN_TIME_UNIVERSE_COLUMNS if column not in membership
    ]
    if missing_columns:
        return {
            "status": "invalid_membership_schema",
            "promotion_eligible": False,
            "required_ticker_count": len(tickers),
            "covered_ticker_count": 0,
            "missing_tickers": tickers,
            "missing_columns": missing_columns,
            "holding_membership_violation_count": None,
            "delisting_treatment_status": "unverified",
            "delisting_evidence_gap_count": None,
        }

    frame = membership.copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["effective_from"] = pd.to_datetime(frame["effective_from"], errors="coerce")
    frame["effective_to"] = pd.to_datetime(frame["effective_to"], errors="coerce")
    covered = sorted(set(tickers).intersection(frame["ticker"]))
    missing_tickers = sorted(set(tickers) - set(covered))
    holding_violations = _holding_membership_violations(weights, frame)
    delisting_gaps = _delisting_evidence_gaps(frame)
    source_gaps = int(
        frame["source"].fillna("").astype(str).str.strip().eq("").sum()
        + frame["source_as_of"].fillna("").astype(str).str.strip().eq("").sum()
    )
    promotion_eligible = not any((missing_tickers, holding_violations, delisting_gaps, source_gaps))
    return {
        "status": "verified" if promotion_eligible else "incomplete_point_in_time_evidence",
        "promotion_eligible": promotion_eligible,
        "required_ticker_count": len(tickers),
        "covered_ticker_count": len(covered),
        "missing_tickers": missing_tickers,
        "holding_membership_violation_count": holding_violations,
        "delisting_treatment_status": "verified" if not delisting_gaps else "incomplete",
        "delisting_evidence_gap_count": delisting_gaps,
        "source_metadata_gap_count": source_gaps,
    }


def build_research_trial_ledger(report_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolidate declared completed trials from persisted research manifests."""

    root = Path(report_root)
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for manifest_path in sorted(root.glob("**/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            coverage_rows.append(
                {
                    "manifest_path": str(manifest_path),
                    "study": "unknown",
                    "status": "manifest_unreadable",
                    "declared_trial_count": 0,
                    "ledger_trial_count": 0,
                }
            )
            continue
        if not isinstance(manifest, dict):
            continue
        study = str(manifest.get("study", manifest_path.parent.name))
        parameters = manifest.get("parameters", {})
        parameter_map = parameters if isinstance(parameters, Mapping) else {}
        candidates, roster_source = _declared_trial_candidates(
            study,
            parameter_map,
            manifest_path.parent,
        )
        governance = manifest.get("research_governance", {})
        governance_map = governance if isinstance(governance, Mapping) else {}
        universe = governance_map.get("point_in_time_universe", {})
        universe_map = universe if isinstance(universe, Mapping) else {}
        for candidate in candidates:
            trial_id = hashlib.sha256(
                "|".join(
                    [
                        study,
                        str(candidate),
                        str(manifest.get("config_sha256", "")),
                        str(manifest_path),
                    ]
                ).encode()
            ).hexdigest()
            rows.append(
                {
                    "experiment_id": trial_id,
                    "trial_id": trial_id,
                    "study": study,
                    "candidate": str(candidate),
                    "parent_experiment_id": str(
                        parameter_map.get("parent_experiment_id", "") or ""
                    ),
                    "hypothesis": str(parameter_map.get("hypothesis", "") or ""),
                    "primary_metric": str(parameter_map.get("primary_metric", "") or ""),
                    "result": str(parameter_map.get("result", "") or ""),
                    "decision": str(parameter_map.get("decision", "") or ""),
                    "decision_reason": str(parameter_map.get("decision_reason", "") or ""),
                    "universe_declaration": _compact_declaration(
                        parameter_map.get("universe", parameter_map.get("tickers", ""))
                    ),
                    "evaluation_period": _evaluation_period(parameter_map),
                    "trial_status": "completed_manifested",
                    "roster_source": roster_source,
                    "generated_at_utc": str(manifest.get("generated_at_utc", "")),
                    "manifest_path": str(manifest_path),
                    "config_sha256": str(manifest.get("config_sha256", "")),
                    "source_tree_sha256": str(
                        (manifest.get("code", {}) or {}).get("source_tree_sha256", "")
                        if isinstance(manifest.get("code", {}), Mapping)
                        else ""
                    ),
                    "automatic_promotion_allowed": bool(
                        manifest.get("automatic_promotion_allowed", False)
                    ),
                    "point_in_time_universe_status": str(
                        universe_map.get("status", "unverified_legacy_manifest")
                    ),
                    "delisting_treatment_status": str(
                        universe_map.get("delisting_treatment_status", "unverified")
                    ),
                }
            )
        coverage_rows.append(
            {
                "manifest_path": str(manifest_path),
                "study": study,
                "status": (
                    "study_level_only"
                    if roster_source == "study_level_fallback"
                    else "declared_roster_indexed"
                ),
                "declared_trial_count": len(candidates),
                "ledger_trial_count": len(candidates),
                "roster_source": roster_source,
            }
        )
    manifested_directories = {
        Path(str(row.get("manifest_path", ""))).parent.resolve()
        for row in coverage_rows
        if str(row.get("manifest_path", ""))
    } if coverage_rows else set()
    for artifact_dir in _unmanifested_artifact_directories(root, manifested_directories):
        coverage_rows.append(
            {
                "manifest_path": "",
                "study": str(artifact_dir.relative_to(root)),
                "status": "artifact_directory_without_manifest",
                "declared_trial_count": 0,
                "ledger_trial_count": 0,
                "roster_source": "unavailable",
            }
        )
    ledger = pd.DataFrame(rows).drop_duplicates("trial_id") if rows else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    return ledger, coverage


def write_research_trial_ledger(
    report_root: str | Path,
    *,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    ledger, coverage = build_research_trial_ledger(report_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "trial_ledger.csv"
    coverage_path = output / "study_coverage.csv"
    summary_path = output / "summary.md"
    ledger.to_csv(ledger_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    manifested_coverage = (
        coverage[coverage["manifest_path"].fillna("").astype(str).ne("")]
        if not coverage.empty
        else coverage
    )
    missing_rosters = (
        int(manifested_coverage["status"].ne("declared_roster_indexed").sum())
        if not manifested_coverage.empty
        else 0
    )
    legacy_universe = (
        int(ledger["point_in_time_universe_status"].ne("verified").sum()) if not ledger.empty else 0
    )
    distinct_studies = int(ledger["study"].nunique()) if not ledger.empty else 0
    distinct_configs = int(ledger["config_sha256"].nunique()) if not ledger.empty else 0
    unmanifested = (
        int(coverage["status"].eq("artifact_directory_without_manifest").sum())
        if not coverage.empty
        else 0
    )
    prospective_required = bool(
        len(ledger) >= RETROSPECTIVE_ADAPTATION_LIMIT or missing_rosters or unmanifested
    )
    lines = [
        "# Research Trial And Universe Governance",
        "",
        f"- Manifested completed trial rows indexed: {len(ledger)}.",
        f"- Study manifests inspected: {len(manifested_coverage)}.",
        f"- Manifests without an explicit candidate roster: {missing_rosters}.",
        f"- Trial rows without verified point-in-time universe evidence: {legacy_universe}.",
        f"- Distinct manifested studies: {distinct_studies}.",
        f"- Distinct manifested config identities: {distinct_configs}.",
        f"- Artifact directories without a manifest: {unmanifested}.",
        (
            "- Retrospective promotion status: `prospective_evidence_required`."
            if prospective_required
            else "- Retrospective promotion status: `eligible_for_human_review`."
        ),
        "",
        (
            "This ledger indexes persisted completed trials. It does not invent interrupted or "
            "unmanifested historical attempts; those remain an explicit completeness gap."
        ),
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return ledger_path, coverage_path, summary_path


def _unmanifested_artifact_directories(
    root: Path,
    manifested_directories: set[Path],
) -> list[Path]:
    directories: list[Path] = []
    if not root.exists():
        return directories
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        if directory.resolve() in manifested_directories:
            continue
        has_research_artifact = any(directory.glob("*.csv")) or any(directory.glob("summary.md"))
        if has_research_artifact:
            directories.append(directory)
    return directories


def _compact_declaration(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(str(item) for item in value)
    return str(value or "")


def _evaluation_period(parameters: Mapping[str, Any]) -> str:
    start = str(parameters.get("start_date", parameters.get("evaluation_start", "")) or "")
    end = str(parameters.get("end_date", parameters.get("evaluation_end", "")) or "")
    return f"{start}..{end}" if start or end else ""


def _holding_membership_violations(weights: pd.DataFrame | None, membership: pd.DataFrame) -> int:
    if weights is None or weights.empty:
        return 0
    violations = 0
    dated_weights = weights.copy()
    dated_weights.index = pd.to_datetime(dated_weights.index, errors="coerce")
    for ticker in dated_weights.columns:
        held_dates = dated_weights.index[
            pd.to_numeric(dated_weights[ticker], errors="coerce") > 1e-12
        ]
        rows = membership[membership["ticker"].eq(str(ticker))]
        for held_date in held_dates:
            active = rows[
                rows["effective_from"].le(held_date)
                & (rows["effective_to"].isna() | rows["effective_to"].ge(held_date))
            ]
            if active.empty:
                violations += 1
    return violations


def _delisting_evidence_gaps(membership: pd.DataFrame) -> int:
    ended = membership[membership["effective_to"].notna()]
    if ended.empty:
        return 0
    included = ended["delisting_return_included"].map(_as_bool)
    source_present = ended["delisting_return_source"].fillna("").astype(str).str.strip().ne("")
    return int((~included | ~source_present).sum())


def _declared_trial_candidates(
    study: str,
    parameters: Mapping[str, Any],
    study_dir: Path,
) -> tuple[list[str], str]:
    for key in ("candidate_names", "candidate_set", "mechanisms", "strategies"):
        values = parameters.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            candidates = [str(value) for value in values if str(value)]
            if candidates:
                return list(dict.fromkeys(candidates)), key
    for filename in ("candidate_roster.csv", "candidates.csv", "strategy_metrics.csv"):
        path = study_dir / filename
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        column = next(
            (
                candidate_column
                for candidate_column in (
                    "strategy",
                    "result_name",
                    "name",
                    "candidate",
                    "mechanism",
                )
                if candidate_column in frame
            ),
            None,
        )
        if column is None:
            continue
        candidates = frame[column].dropna().astype(str).drop_duplicates().tolist()
        if candidates:
            return candidates, filename
    strategy = parameters.get("strategy")
    if strategy:
        return [str(strategy)], "strategy"
    return [study], "study_level_fallback"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
