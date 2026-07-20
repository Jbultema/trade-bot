from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False, ttl=60)
def read_csv_artifact(path: str | Path) -> pd.DataFrame:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return pd.DataFrame()
    return pd.read_csv(artifact_path)


def pbo_frames(report_dir: str | Path = "reports/pbo_diagnostics") -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "summary": read_csv_artifact(root / "pbo_summary.csv"),
        "selection": read_csv_artifact(root / "pbo_strategy_selection.csv"),
        "stats": read_csv_artifact(root / "pbo_strategy_stats.csv"),
    }


def leadership_frames(report_dir: str | Path = "reports/leadership_diagnostics") -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "summary": read_csv_artifact(root / "leadership_summary.csv"),
        "impairment": read_csv_artifact(root / "leadership_impairment.csv"),
        "router": read_csv_artifact(root / "walk_forward_router_comparison.csv"),
    }


def prebreak_hindsight_frames(
    report_dir: str | Path = "reports/prebreak_hindsight",
) -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "snapshot_signal_panel": read_csv_artifact(root / "snapshot_signal_panel.csv"),
        "signal_predictiveness_rank": read_csv_artifact(
            root / "signal_predictiveness_rank.csv"
        ),
        "action_timing": read_csv_artifact(root / "action_timing.csv"),
        "staged_risk_behavior": read_csv_artifact(root / "staged_risk_behavior.csv"),
        "late_trigger_mesh": read_csv_artifact(root / "late_trigger_mesh.csv"),
        "hard_defense_attribution": read_csv_artifact(
            root / "hard_defense_attribution.csv"
        ),
        "policy_variant_results": read_csv_artifact(root / "policy_variant_results.csv"),
        "current_best_signal_readout": read_csv_artifact(
            root / "current_best_signal_readout.csv"
        ),
    }


def defensive_signal_audit_frames(
    report_dir: str | Path = "reports/defensive_signal_audit",
) -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "current_defensive_exposure": read_csv_artifact(
            root / "current_defensive_exposure.csv"
        ),
        "summary": read_csv_artifact(root / "defensive_signal_summary.csv"),
        "scorecards": read_csv_artifact(root / "defensive_signal_scorecards.csv"),
    }


def cycle_tracker_frames(report_dir: str | Path = "reports/cycle_tracker") -> dict[str, pd.DataFrame]:
    root = Path(report_dir)
    return {
        "phase_probabilities": read_csv_artifact(root / "cycle_phase_probabilities.csv"),
        "transition_forecast": read_csv_artifact(root / "cycle_transition_forecast.csv"),
        "evidence": read_csv_artifact(root / "cycle_evidence_components.csv"),
        "path_state_history": read_csv_artifact(root / "cycle_path_state_history.csv"),
        "path_transition_forecast": read_csv_artifact(
            root / "cycle_path_transition_forecast.csv"
        ),
        "candidate_scores": read_csv_artifact(root / "cycle_candidate_scores.csv"),
        "phase_candidate_frontier": read_csv_artifact(
            root / "cycle_phase_candidate_frontier.csv"
        ),
        "validation_metrics": read_csv_artifact(root / "cycle_validation_metrics.csv"),
        "path_validation_metrics": read_csv_artifact(
            root / "cycle_path_validation_metrics.csv"
        ),
        "validation_observations": read_csv_artifact(
            root / "cycle_validation_observations.csv"
        ),
        "phase_reliability": read_csv_artifact(root / "cycle_phase_reliability.csv"),
        "path_reliability": read_csv_artifact(root / "cycle_path_reliability.csv"),
        "crisis_playback": read_csv_artifact(root / "cycle_crisis_playback.csv"),
    }
