from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from trade_bot.config import BotConfig, StrategyConfig
from trade_bot.research.artifact_provenance import research_source_tree_sha256
from trade_bot.research.experiment_replay import verify_experiment_library

DEFAULT_MODEL_EVIDENCE_OUTPUT_DIR = Path("reports/model_evidence_audit")
REFERENCE_TYPES = {"buy_hold", "fixed_allocation"}
ML_MARKERS = ("_ml_", "extra_trees", "random_forest", "xgboost")
ADAPTIVE_TYPES = {
    "absolute_momentum",
    "relative_momentum",
    "dual_momentum",
    "ai_risk_cycle_overlay",
}


@dataclass(frozen=True)
class ModelEvidenceAudit:
    slate: pd.DataFrame
    evidence_matrix: pd.DataFrame
    requirement_matrix: pd.DataFrame
    prospective_registry: pd.DataFrame
    summary: str


def build_relevant_model_slate(
    config: BotConfig,
    scorecards: pd.DataFrame,
    *,
    top_n: int = 12,
) -> pd.DataFrame:
    """Select configured, reference, return-leading, utility-leading, and Pareto models."""

    if scorecards.empty:
        rows = [
            _configured_slate_row(name, strategy, config)
            for name, strategy in config.strategies.items()
        ]
        return pd.DataFrame(rows).sort_values(["model_class", "strategy"]).reset_index(drop=True)

    latest = (
        scorecards.sort_values(["iteration", "iteration_rank"], na_position="last")
        .drop_duplicates("strategy", keep="last")
        .copy()
    )
    latest["cagr"] = pd.to_numeric(latest.get("cagr"), errors="coerce")
    utility_column = _first_present(
        latest,
        (
            "growth_constrained_utility_score",
            "after_tax_growth_constrained_utility_score",
            "promotion_score",
        ),
    )
    utility = (
        pd.to_numeric(latest[utility_column], errors="coerce")
        if utility_column is not None
        else pd.Series(float("nan"), index=latest.index)
    )
    pareto = pd.Series(False, index=latest.index)
    for column in (
        "is_growth_pareto_efficient",
        "is_pareto_efficient",
        "pareto_frontier",
    ):
        if column in latest:
            pareto |= latest[column].fillna(False).astype(bool)

    configured = set(config.strategies)
    top_cagr = set(latest.nlargest(top_n, "cagr")["strategy"])
    top_utility = set(latest.loc[utility.nlargest(top_n).index, "strategy"])
    pareto_names = set(latest.loc[pareto, "strategy"])
    selected = configured | top_cagr | top_utility | pareto_names

    rows: list[dict[str, object]] = []
    by_name = latest.set_index("strategy", drop=False)
    for name in sorted(selected):
        if name in by_name.index:
            row = by_name.loc[name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            strategy = config.strategies.get(name)
            strategy_type = (
                strategy.type if strategy is not None else _strategy_type_from_row(row)
            )
            rows.append(
                {
                    "strategy": name,
                    "model_class": classify_model(name, strategy_type, row),
                    "configured": name in configured,
                    "primary": name == config.primary_strategy,
                    "reference": _is_reference(name, strategy_type),
                    "top_cagr": name in top_cagr,
                    "top_utility": name in top_utility,
                    "pareto": name in pareto_names,
                    "iteration": _optional_int(row.get("iteration")),
                    "cagr": _optional_float(row.get("cagr")),
                    "max_drawdown": _optional_float(row.get("max_drawdown")),
                    "utility": _optional_float(row.get(utility_column))
                    if utility_column is not None
                    else None,
                    "strategy_type": strategy_type,
                    "strategy_json": str(row.get("strategy_json", "") or ""),
                }
            )
        else:
            rows.append(_configured_slate_row(name, config.strategies[name], config))
    return pd.DataFrame(rows).sort_values(
        ["primary", "reference", "top_cagr", "strategy"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def classify_model(name: str, strategy_type: str, row: pd.Series | None = None) -> str:
    lowered = name.lower()
    if _is_reference(name, strategy_type):
        return "reference"
    if any(marker in lowered for marker in ML_MARKERS):
        return "machine_learning"
    if row is not None and any(
        _nonempty(row.get(column))
        for column in ("future_state_model_json", "strategy_drawdown_model_json")
    ):
        return "machine_learning"
    if "risk_repair" in lowered or "overlay" in lowered:
        return "adaptive_risk_overlay"
    if strategy_type in ADAPTIVE_TYPES:
        return "adaptive_selector"
    return "rules_based"


def build_model_evidence_audit(
    config: BotConfig,
    scorecards: pd.DataFrame,
    *,
    experiment_root: str | Path,
    report_root: str | Path = "reports",
    selection_cutoff: date,
    top_n: int = 12,
) -> ModelEvidenceAudit:
    experiment_path = Path(experiment_root)
    report_path = Path(report_root)
    slate = build_relevant_model_slate(config, scorecards, top_n=top_n)
    library_verification = verify_experiment_library(experiment_path, config=config)
    library_status = str(library_verification.get("status", "unverified"))
    evidence = _build_evidence_matrix(
        slate,
        scorecards,
        experiment_path=experiment_path,
        report_path=report_path,
        library_status=library_status,
    )
    requirements = _build_requirement_matrix(evidence)
    prospective = _build_prospective_registry(
        slate,
        selection_cutoff=selection_cutoff,
    )
    summary = _summary_markdown(
        slate,
        evidence,
        requirements,
        prospective,
        library_status=library_status,
    )
    return ModelEvidenceAudit(
        slate=slate,
        evidence_matrix=evidence,
        requirement_matrix=requirements,
        prospective_registry=prospective,
        summary=summary,
    )


def write_model_evidence_audit(
    audit: ModelEvidenceAudit,
    output_dir: str | Path = DEFAULT_MODEL_EVIDENCE_OUTPUT_DIR,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "slate": output / "relevant_model_slate.csv",
        "evidence": output / "strategy_evidence_matrix.csv",
        "requirements": output / "test_requirement_matrix.csv",
        "prospective": output / "prospective_registry.csv",
        "summary": output / "summary.md",
        "manifest": output / "manifest.json",
    }
    audit.slate.to_csv(artifacts["slate"], index=False)
    audit.evidence_matrix.to_csv(artifacts["evidence"], index=False)
    audit.requirement_matrix.to_csv(artifacts["requirements"], index=False)
    audit.prospective_registry.to_csv(artifacts["prospective"], index=False)
    artifacts["summary"].write_text(audit.summary, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "study": "relevant_model_evidence_audit",
        "status": "complete",
        "automatic_promotion_allowed": False,
        "source_tree_sha256": research_source_tree_sha256(),
        "strategy_count": int(len(audit.slate)),
        "selection_oos_verified_count": int(
            audit.evidence_matrix["selection_oos_status"].eq("verified").sum()
        ),
        "prospective_candidate_count": int(len(audit.prospective_registry)),
        "artifacts": {
            key: {
                "path": path.name,
                "sha256": _file_sha256(path),
            }
            for key, path in artifacts.items()
            if key != "manifest"
        },
    }
    artifacts["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifacts


def _build_evidence_matrix(
    slate: pd.DataFrame,
    scorecards: pd.DataFrame,
    *,
    experiment_path: Path,
    report_path: Path,
    library_status: str,
) -> pd.DataFrame:
    latest = (
        scorecards.sort_values(["iteration", "iteration_rank"], na_position="last")
        .drop_duplicates("strategy", keep="last")
        .set_index("strategy", drop=False)
        if not scorecards.empty
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for _, item in slate.iterrows():
        name = str(item["strategy"])
        row = latest.loc[name] if not latest.empty and name in latest.index else pd.Series(dtype=object)
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        iteration = _optional_int(item.get("iteration"))
        iteration_root = (
            experiment_path / f"iteration_{iteration:02d}" if iteration is not None else None
        )
        model_class = str(item["model_class"])
        is_reference = bool(item["reference"])
        specialized = _specialized_evidence(name, report_path)
        raw_oos = str(row.get("oos_evidence_status", "") or "")
        selection_oos = (
            "not_required_reference"
            if is_reference
            else ("verified" if raw_oos.startswith("verified_") else "missing")
        )
        rows.append(
            {
                "strategy": name,
                "model_class": model_class,
                "configured": bool(item["configured"]),
                "primary": bool(item["primary"]),
                "reference": is_reference,
                "current_source_replay": "pass"
                if library_status == "current_complete"
                else "stale_or_missing",
                "full_period_backtest": _artifact_status(iteration_root, "metrics.csv"),
                "rolling_window_analysis": _artifact_status(iteration_root, "window_summary.csv"),
                "temporal_walk_forward": _metric_or_artifact_status(
                    row,
                    "walk_forward_positive_rate",
                    iteration_root,
                    "walk_forward_summary.csv",
                ),
                "regime_and_crisis_analysis": _metric_or_artifact_status(
                    row,
                    "worst_regime_return",
                    iteration_root,
                    "regime_metrics.csv",
                ),
                "operability_and_turnover": _metric_or_artifact_status(
                    row,
                    "operability_score",
                    iteration_root,
                    "operability_metrics.csv",
                ),
                "transition_analysis": _metric_or_artifact_status(
                    row,
                    "transition_regime_hit_rate",
                    iteration_root,
                    "transition_metrics.csv",
                ),
                "execution_sensitivity": specialized["execution_sensitivity"],
                "selection_overfit_pbo": specialized["pbo"],
                "bootstrap_sequence_risk": specialized["bootstrap"],
                "selection_oos_status": selection_oos,
                "selection_oos_method": raw_oos or "undeclared",
                "prospective_shadow_status": "not_required_reference"
                if is_reference
                else "frozen_pending",
                "evidence_read": _evidence_read(
                    model_class=model_class,
                    is_reference=is_reference,
                    library_status=library_status,
                    selection_oos=selection_oos,
                    specialized=specialized,
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_requirement_matrix(evidence: pd.DataFrame) -> pd.DataFrame:
    test_names = (
        "current_source_replay",
        "full_period_backtest",
        "rolling_window_analysis",
        "temporal_walk_forward",
        "regime_and_crisis_analysis",
        "operability_and_turnover",
        "transition_analysis",
        "execution_sensitivity",
        "selection_overfit_pbo",
        "bootstrap_sequence_risk",
        "selection_oos_status",
        "prospective_shadow_status",
    )
    rows: list[dict[str, object]] = []
    for _, model in evidence.iterrows():
        model_class = str(model["model_class"])
        for test_name in test_names:
            required = _test_requirement(model_class, test_name)
            observed = str(model[test_name])
            rows.append(
                {
                    "strategy": model["strategy"],
                    "model_class": model_class,
                    "test": test_name,
                    "requirement": required,
                    "observed_status": observed,
                    "satisfied": _requirement_satisfied(required, observed),
                }
            )
    return pd.DataFrame(rows)


def _build_prospective_registry(
    slate: pd.DataFrame,
    *,
    selection_cutoff: date,
) -> pd.DataFrame:
    evaluation_start = selection_cutoff + timedelta(days=1)
    rows: list[dict[str, object]] = []
    for _, row in slate.loc[~slate["reference"]].iterrows():
        definition = str(row.get("strategy_json", "") or "")
        if not definition:
            definition = json.dumps(
                {
                    "strategy": row["strategy"],
                    "configured": bool(row["configured"]),
                    "strategy_type": row["strategy_type"],
                },
                sort_keys=True,
            )
        rows.append(
            {
                "strategy": row["strategy"],
                "model_class": row["model_class"],
                "selection_cutoff": selection_cutoff.isoformat(),
                "evaluation_start": evaluation_start.isoformat(),
                "method": "prospective_shadow",
                "parameters_frozen_before_evaluation": True,
                "evaluation_used_for_model_selection": False,
                "primary_metric": "growth_constrained_utility_score",
                "primary_metric_pre_registered": True,
                "minimum_observations": 252,
                "minimum_calendar_span_days": 365,
                "benchmark": "buy_hold_qqq",
                "status": "frozen_pending",
                "definition_sha256": hashlib.sha256(definition.encode("utf-8")).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def _summary_markdown(
    slate: pd.DataFrame,
    evidence: pd.DataFrame,
    requirements: pd.DataFrame,
    prospective: pd.DataFrame,
    *,
    library_status: str,
) -> str:
    gaps = requirements.loc[
        requirements["requirement"].eq("required") & ~requirements["satisfied"]
    ]
    gap_counts = gaps.groupby("test").size().sort_values(ascending=False)
    class_counts = slate.groupby("model_class").size().sort_values(ascending=False)
    lines = [
        "# Relevant Model Evidence Audit",
        "",
        "Status: audit complete; no strategy is automatically promoted.",
        "",
        "## Bottom line",
        "",
        f"- Relevant slate: {len(slate)} strategies.",
        f"- Configured/reference strategies: {int(slate['configured'].sum())}.",
        f"- Valid selection-OOS evidence: {int(evidence['selection_oos_status'].eq('verified').sum())}.",
        f"- Prospective shadows frozen: {len(prospective)}.",
        f"- Canonical comparison library: `{library_status}`.",
        (
            "- Historical walk-forward, regime, bootstrap, and PBO evidence remains useful "
            "robustness evidence, but it is not relabeled as selection OOS."
        ),
        (
            "- Static references are OOS anchors rather than promoted forecasts; every adaptive "
            "or learned model requires verified selection OOS before promotion."
        ),
        "",
        "## Slate by model class",
        "",
    ]
    lines.extend(f"- `{name}`: {int(count)}" for name, count in class_counts.items())
    lines.extend(["", "## Required-test gaps", ""])
    if gap_counts.empty:
        lines.append("- None.")
    else:
        lines.extend(f"- `{name}`: {int(count)} strategies." for name, count in gap_counts.items())
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "A stale current-source replay is a comparability blocker, not evidence that the "
                "historical result was fabricated. It means the old metric cannot be mixed with "
                "new-source results in a ranking."
            ),
            (
                "The prospective registry freezes candidate definitions and the primary metric. "
                "It becomes valid OOS evidence only after at least 252 observations and one calendar "
                "year have elapsed without using the shadow outcomes to retune the candidate."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _configured_slate_row(
    name: str,
    strategy: StrategyConfig,
    config: BotConfig,
) -> dict[str, object]:
    strategy_json = json.dumps(strategy.model_dump(mode="json"), sort_keys=True)
    return {
        "strategy": name,
        "model_class": classify_model(name, strategy.type),
        "configured": True,
        "primary": name == config.primary_strategy,
        "reference": _is_reference(name, strategy.type),
        "top_cagr": False,
        "top_utility": False,
        "pareto": False,
        "iteration": None,
        "cagr": None,
        "max_drawdown": None,
        "utility": None,
        "strategy_type": strategy.type,
        "strategy_json": strategy_json,
    }


def _specialized_evidence(name: str, report_path: Path) -> dict[str, str]:
    generic_stress = report_path / "model_evidence_audit" / "adaptive_ml_stress"
    generic_pbo = report_path / "model_evidence_audit" / "pbo_adaptive_ml"
    generic_execution = _tabular_artifact_covers(
        generic_stress / "execution_sensitivity.csv",
        name,
        strategy_columns=("strategy", "result_name"),
    )
    generic_bootstrap = _tabular_artifact_covers(
        generic_stress / "sequence_bootstrap.csv",
        name,
        strategy_columns=("strategy", "result_name"),
    )
    generic_pbo_status = _tabular_artifact_covers(
        generic_pbo / "pbo_candidate_audit.csv",
        name,
        strategy_columns=("strategy", "result_name"),
        required_status=("evaluated", "reconstructed"),
    )
    if not name.startswith("i111"):
        return {
            "execution_sensitivity": generic_execution,
            "pbo": generic_pbo_status,
            "bootstrap": generic_bootstrap,
        }
    candidates = (
        report_path / "i111_adversarial_validation",
        report_path / "i111_adversarial_validation_active",
        report_path / "i111_selector_transition_repair",
        report_path / "i111_execution_smoothing",
    )
    return {
        "execution_sensitivity": _best_status(
            generic_execution,
            _any_artifact(candidates, "execution_sensitivity.csv"),
        ),
        "pbo": _best_status(
            generic_pbo_status,
            _any_artifact(candidates, "candidate_pbo_summary.csv", "pbo_summary.csv"),
        ),
        "bootstrap": _best_status(
            generic_bootstrap,
            _any_artifact(candidates, "sequence_bootstrap.csv", "block_bootstrap.csv"),
        ),
    }


def _test_requirement(model_class: str, test_name: str) -> str:
    if test_name in {
        "current_source_replay",
        "full_period_backtest",
        "rolling_window_analysis",
        "regime_and_crisis_analysis",
    }:
        return "required"
    if model_class == "reference":
        return "not_applicable"
    if test_name in {
        "temporal_walk_forward",
        "operability_and_turnover",
        "selection_oos_status",
        "prospective_shadow_status",
    }:
        return "required"
    if model_class in {"adaptive_selector", "adaptive_risk_overlay", "machine_learning"}:
        return "required"
    return "recommended"


def _requirement_satisfied(requirement: str, status: str) -> bool:
    if requirement == "not_applicable":
        return True
    return status in {
        "pass",
        "verified",
        "not_required_reference",
        "frozen_pending",
    }


def _evidence_read(
    *,
    model_class: str,
    is_reference: bool,
    library_status: str,
    selection_oos: str,
    specialized: dict[str, str],
) -> str:
    if library_status != "current_complete":
        return "historical_metrics_stale_under_current_source"
    if is_reference:
        return "current_reference_anchor"
    if selection_oos != "verified":
        return "robustness_evidence_only_pending_selection_oos"
    if model_class in {"adaptive_selector", "adaptive_risk_overlay", "machine_learning"} and any(
        value != "pass" for value in specialized.values()
    ):
        return "selection_oos_verified_specialized_stress_gaps"
    return "promotion_evidence_complete"


def _artifact_status(root: Path | None, filename: str) -> str:
    return "pass" if root is not None and (root / filename).is_file() else "missing"


def _metric_or_artifact_status(
    row: pd.Series,
    metric: str,
    root: Path | None,
    filename: str,
) -> str:
    return (
        "pass"
        if (metric in row and pd.notna(row.get(metric))) or _artifact_status(root, filename) == "pass"
        else "missing"
    )


def _any_artifact(roots: tuple[Path, ...], *filenames: str) -> str:
    return (
        "pass"
        if any((root / filename).is_file() for root in roots for filename in filenames)
        else "missing"
    )


def _tabular_artifact_covers(
    path: Path,
    strategy: str,
    *,
    strategy_columns: tuple[str, ...],
    required_status: tuple[str, ...] = (),
) -> str:
    if not path.is_file():
        return "missing"
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return "missing"
    strategy_column = next((column for column in strategy_columns if column in frame), None)
    if strategy_column is None:
        return "missing"
    rows = frame.loc[frame[strategy_column].astype(str).eq(strategy)]
    if rows.empty:
        return "missing"
    if required_status and "status" in rows:
        statuses = set(rows["status"].dropna().astype(str))
        if not statuses.intersection(required_status):
            return "missing"
    return "pass"


def _best_status(*statuses: str) -> str:
    return "pass" if "pass" in statuses else "missing"


def _strategy_type_from_row(row: pd.Series) -> str:
    raw = str(row.get("strategy_json", "") or "")
    try:
        decoded = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return "unknown"
    return str(decoded.get("type", "unknown")) if isinstance(decoded, dict) else "unknown"


def _is_reference(name: str, strategy_type: str) -> bool:
    return (
        strategy_type in REFERENCE_TYPES
        or name.startswith("buy_hold_")
        or name.startswith("i41_ref_")
    )


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame), None)


def _nonempty(value: object) -> bool:
    return bool(str(value or "").strip()) and str(value).lower() != "nan"


def _optional_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def _optional_int(value: object) -> int | None:
    numeric = _optional_float(value)
    return None if numeric is None else int(numeric)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
