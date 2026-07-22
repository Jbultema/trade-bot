from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from trade_bot.config import load_config
from trade_bot.DEFAULTS import DEFAULT_CONFIG_PATH
from trade_bot.research.artifact_provenance import (
    research_config_sha256,
    research_source_tree_sha256,
    verify_research_manifest,
)


@st.cache_data(show_spinner=False, ttl=60)
def read_csv_artifact(path: str | Path) -> pd.DataFrame:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return pd.DataFrame()
    return pd.read_csv(artifact_path)


@st.cache_data(show_spinner=False, ttl=60)
def read_json_artifact(path: str | Path) -> dict[str, object]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {}
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def pbo_frames(report_dir: str | Path = "reports/pbo_diagnostics") -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "summary": _read_current_manifested_csv(root, "pbo_summary.csv"),
        "selection": _read_current_manifested_csv(root, "pbo_strategy_selection.csv"),
        "stats": _read_current_manifested_csv(root, "pbo_strategy_stats.csv"),
    }


def leadership_frames(
    report_dir: str | Path = "reports/leadership_diagnostics",
) -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "summary": _read_current_manifested_csv(root, "walk_forward_router_summary.csv"),
        "impairment": _read_current_manifested_csv(root, "leadership_impairment.csv"),
        "router": _read_current_manifested_csv(root, "walk_forward_router_comparison.csv"),
    }


def prebreak_hindsight_frames(
    report_dir: str | Path = "reports/prebreak_hindsight",
) -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "snapshot_signal_panel": read_csv_artifact(root / "snapshot_signal_panel.csv"),
        "signal_predictiveness_rank": read_csv_artifact(root / "signal_predictiveness_rank.csv"),
        "action_timing": read_csv_artifact(root / "action_timing.csv"),
        "staged_risk_behavior": read_csv_artifact(root / "staged_risk_behavior.csv"),
        "late_trigger_mesh": read_csv_artifact(root / "late_trigger_mesh.csv"),
        "hard_defense_attribution": read_csv_artifact(root / "hard_defense_attribution.csv"),
        "policy_variant_results": read_csv_artifact(root / "policy_variant_results.csv"),
        "current_best_signal_readout": read_csv_artifact(root / "current_best_signal_readout.csv"),
    }


def defensive_signal_audit_frames(
    report_dir: str | Path = "reports/defensive_signal_audit",
) -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "current_defensive_exposure": read_csv_artifact(root / "current_defensive_exposure.csv"),
        "summary": read_csv_artifact(root / "defensive_signal_summary.csv"),
        "scorecards": read_csv_artifact(root / "defensive_signal_scorecards.csv"),
    }


def cycle_tracker_frames(
    report_dir: str | Path = "reports/cycle_tracker",
) -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "phase_probabilities": read_csv_artifact(root / "cycle_phase_probabilities.csv"),
        "transition_forecast": read_csv_artifact(root / "cycle_transition_forecast.csv"),
        "evidence": read_csv_artifact(root / "cycle_evidence_components.csv"),
        "path_state_history": read_csv_artifact(root / "cycle_path_state_history.csv"),
        "path_transition_forecast": read_csv_artifact(root / "cycle_path_transition_forecast.csv"),
        "candidate_scores": read_csv_artifact(root / "cycle_candidate_scores.csv"),
        "phase_candidate_frontier": read_csv_artifact(root / "cycle_phase_candidate_frontier.csv"),
        "validation_metrics": read_csv_artifact(root / "cycle_validation_metrics.csv"),
        "path_validation_metrics": read_csv_artifact(root / "cycle_path_validation_metrics.csv"),
        "validation_observations": read_csv_artifact(root / "cycle_validation_observations.csv"),
        "phase_reliability": read_csv_artifact(root / "cycle_phase_reliability.csv"),
        "path_reliability": read_csv_artifact(root / "cycle_path_reliability.csv"),
        "crisis_playback": read_csv_artifact(root / "cycle_crisis_playback.csv"),
    }


def i111_evidence_frames(report_root: str | Path = "reports") -> dict[str, pd.DataFrame]:
    root = Path(report_root)
    return {
        "native_metrics": _read_current_manifested_csv(
            root / "native_i111_risk_repair", "strategy_metrics.csv"
        ),
        "adversarial_robustness": _read_current_manifested_csv(
            root / "i111_adversarial_validation", "robustness_summary.csv"
        ),
        "adversarial_gaps": _read_current_manifested_csv(
            root / "i111_adversarial_validation", "gap_audit.csv"
        ),
        "execution_mechanisms": _read_current_manifested_csv(
            root / "i111_execution_hardening", "mechanism_summary.csv"
        ),
        "smoothing_gates": _smoothing_gate_frame(
            root / "i111_execution_smoothing",
            "promotion_gates.csv",
        ),
        "smoothing_summary": _read_current_manifested_csv(
            root / "i111_execution_smoothing", "schedule_summary.csv"
        ),
        "smoothing_pbo": _read_current_manifested_csv(
            root / "i111_execution_smoothing", "pbo_summary.csv"
        ),
        "qc_headline": _read_current_manifested_csv(
            root / "backtest_qc_i111_native", "headline.csv"
        ),
        "manifests": i111_manifest_index(root),
    }


def _read_current_manifested_csv(root: Path, filename: str) -> pd.DataFrame:
    manifest = read_json_artifact(root / "manifest.json")
    verification = verify_research_manifest(
        root / "manifest.json",
        current_source_tree_sha256=_current_research_source_tree_sha256(),
    )
    if (
        verification.get("artifact_integrity_status") != "verified"
        or verification.get("source_tree_status") != "current"
        or manifest.get("config_sha256") != _current_research_config_sha256()
    ):
        return pd.DataFrame()
    return read_csv_artifact(root / filename)


def i111_manifest_index(report_root: str | Path = "reports") -> pd.DataFrame:
    root = Path(report_root)
    report_dirs = (
        "native_i111_risk_repair",
        "i111_risk_repair",
        "i111_orthogonal_search",
        "i111_frontier_search",
        "i111_adversarial_validation",
        "i111_execution_hardening",
        "i111_execution_smoothing",
    )
    rows: list[dict[str, object]] = []
    current_source_hash = _current_research_source_tree_sha256()
    for report_dir in report_dirs:
        manifest_path = root / report_dir / "manifest.json"
        manifest = read_json_artifact(manifest_path)
        if not manifest:
            continue
        price_input = manifest.get("price_input", {})
        code = manifest.get("code", {})
        verification = verify_research_manifest(
            manifest_path,
            current_source_tree_sha256=current_source_hash,
        )
        rows.append(
            {
                "study": manifest.get("study", report_dir),
                "generated_at_utc": manifest.get("generated_at_utc", ""),
                "market_date": (
                    price_input.get("market_date", "") if isinstance(price_input, dict) else ""
                ),
                "git_sha": code.get("git_sha", "") if isinstance(code, dict) else "",
                "git_dirty": code.get("git_dirty", "") if isinstance(code, dict) else "",
                "config_sha256": manifest.get("config_sha256", ""),
                "research_status": manifest.get("research_status", ""),
                "automatic_promotion_allowed": manifest.get(
                    "automatic_promotion_allowed",
                    False,
                ),
                **verification,
                "manifest_path": str(manifest_path),
            }
        )
    return pd.DataFrame(rows)


def _smoothing_gate_frame(root: Path, filename: str) -> pd.DataFrame:
    frame = _read_current_manifested_csv(root, filename)
    if frame.empty:
        return frame
    output = frame.copy()
    if "local_pbo" in output and "family_pbo_gate" not in output:
        output = output.rename(columns={"local_pbo": "family_pbo_gate"})
    if "pbo_scope" not in output:
        output["pbo_scope"] = "family_15_strategies_70_splits"
    return output


@st.cache_data(show_spinner=False, ttl=60)
def _current_research_source_tree_sha256() -> str:
    return research_source_tree_sha256()


@st.cache_data(show_spinner=False, ttl=60)
def _current_research_config_sha256() -> str:
    return research_config_sha256(load_config(DEFAULT_CONFIG_PATH))
