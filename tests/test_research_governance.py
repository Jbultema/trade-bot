from __future__ import annotations

import json

import pandas as pd

from trade_bot.research.research_governance import (
    audit_out_of_sample_evidence,
    audit_point_in_time_universe,
    build_research_trial_ledger,
    write_research_trial_ledger,
)


def test_oos_audit_fails_closed_when_missing_or_same_history() -> None:
    missing = audit_out_of_sample_evidence(None)
    same_history = audit_out_of_sample_evidence(
        {
            "method": "retrospective_same_history",
            "completed": True,
            "parameters_frozen_before_evaluation": False,
            "evaluation_used_for_model_selection": True,
            "primary_metric_pre_registered": True,
            "selection_cutoff": "2026-07-24",
            "evaluation_start": "2015-01-01",
            "evaluation_end": "2026-07-24",
            "observation_count": 500,
            "minimum_observations": 252,
        }
    )

    assert missing["status"] == "missing_oos_evidence"
    assert missing["promotion_eligible"] is False
    assert same_history["status"] == "invalid_or_incomplete_oos_evidence"
    assert same_history["promotion_eligible"] is False
    assert "unsupported_or_retrospective_method" in same_history["gaps"]
    assert "evaluation_used_for_model_selection" in same_history["gaps"]
    assert "evaluation_must_start_after_selection_cutoff" in same_history["gaps"]


def test_oos_audit_accepts_sealed_holdout_only_after_freeze() -> None:
    audit = audit_out_of_sample_evidence(
        {
            "method": "sealed_holdout",
            "completed": True,
            "parameters_frozen_before_evaluation": True,
            "evaluation_used_for_model_selection": False,
            "primary_metric_pre_registered": True,
            "selection_cutoff": "2024-12-31",
            "evaluation_start": "2025-01-02",
            "evaluation_end": "2026-06-30",
            "observation_count": 375,
            "minimum_observations": 252,
        }
    )

    assert audit["status"] == "verified_sealed_holdout"
    assert audit["promotion_eligible"] is True
    assert audit["gaps"] == []


def test_nested_walk_forward_requires_untouched_outer_predictions() -> None:
    evidence = {
        "method": "nested_walk_forward",
        "completed": True,
        "parameters_frozen_before_evaluation": True,
        "evaluation_used_for_model_selection": False,
        "primary_metric_pre_registered": True,
        "selection_cutoff": "2014-12-31",
        "evaluation_start": "2015-01-02",
        "evaluation_end": "2026-06-30",
        "observation_count": 2800,
        "minimum_observations": 1000,
        "inner_model_selection_only": True,
        "outer_predictions_untouched": False,
    }

    blocked = audit_out_of_sample_evidence(evidence)
    evidence["outer_predictions_untouched"] = True
    verified = audit_out_of_sample_evidence(evidence)

    assert "outer_predictions_not_declared_untouched" in blocked["gaps"]
    assert blocked["promotion_eligible"] is False
    assert verified["status"] == "verified_nested_walk_forward"
    assert verified["promotion_eligible"] is True


def test_incomplete_prospective_shadow_is_not_oos_evidence_yet() -> None:
    audit = audit_out_of_sample_evidence(
        {
            "method": "prospective_shadow",
            "completed": False,
            "parameters_frozen_before_evaluation": True,
            "evaluation_used_for_model_selection": False,
            "primary_metric_pre_registered": True,
            "selection_cutoff": "2026-07-24",
            "evaluation_start": "2026-07-25",
            "evaluation_end": None,
            "observation_count": 0,
            "minimum_observations": 63,
        }
    )

    assert audit["promotion_eligible"] is False
    assert "evaluation_not_completed" in audit["gaps"]
    assert "positive_observation_count_required" in audit["gaps"]
    assert "minimum_observation_count_not_met" not in audit["gaps"]
    assert "evaluation_end_required" in audit["gaps"]


def test_completed_holdout_must_meet_preregistered_minimum_sample() -> None:
    audit = audit_out_of_sample_evidence(
        {
            "method": "sealed_holdout",
            "completed": True,
            "parameters_frozen_before_evaluation": True,
            "evaluation_used_for_model_selection": False,
            "primary_metric_pre_registered": True,
            "selection_cutoff": "2025-12-31",
            "evaluation_start": "2026-01-02",
            "evaluation_end": "2026-02-02",
            "observation_count": 20,
            "minimum_observations": 63,
        }
    )

    assert audit["promotion_eligible"] is False
    assert "minimum_observation_count_not_met" in audit["gaps"]


def test_point_in_time_universe_audit_fails_closed_without_membership() -> None:
    prices = pd.DataFrame(
        {"QQQ": [100.0, 101.0], "NVDA": [50.0, 51.0]},
        index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
    )

    audit = audit_point_in_time_universe(prices)

    assert audit["status"] == "missing_point_in_time_membership"
    assert audit["promotion_eligible"] is False
    assert audit["missing_tickers"] == ["NVDA", "QQQ"]
    assert audit["delisting_treatment_status"] == "unverified"


def test_point_in_time_universe_audit_checks_hold_dates_and_delistings() -> None:
    index = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    prices = pd.DataFrame({"AAA": [10.0, 9.0, 8.0]}, index=index)
    weights = pd.DataFrame({"AAA": [0.0, 1.0, 1.0]}, index=index)
    membership = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "effective_from": ["2025-01-02"],
            "effective_to": ["2025-01-03"],
            "source": ["exchange_constituent_file"],
            "source_as_of": ["2025-01-03"],
            "delisting_return_included": [True],
            "delisting_return_source": ["vendor_delisting_file"],
        }
    )

    audit = audit_point_in_time_universe(prices, membership, weights=weights)

    assert audit["status"] == "verified"
    assert audit["promotion_eligible"] is True
    assert audit["holding_membership_violation_count"] == 0
    assert audit["delisting_treatment_status"] == "verified"


def test_trial_ledger_indexes_declared_candidates_without_inventing_attempts(tmp_path) -> None:
    study = tmp_path / "reports" / "fixed_study"
    study.mkdir(parents=True)
    (study / "manifest.json").write_text(
        json.dumps(
            {
                "study": "fixed_study",
                "generated_at_utc": "2026-07-21T00:00:00+00:00",
                "config_sha256": "abc",
                "code": {"source_tree_sha256": "def"},
                "automatic_promotion_allowed": False,
                "parameters": {"candidate_set": ["raw", "replacement_guard"]},
                "research_governance": {
                    "point_in_time_universe": {
                        "status": "missing_point_in_time_membership",
                        "delisting_treatment_status": "unverified",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    ledger, coverage = build_research_trial_ledger(tmp_path / "reports")

    assert set(ledger["candidate"]) == {"raw", "replacement_guard"}
    assert ledger["experiment_id"].equals(ledger["trial_id"])
    assert set(ledger["trial_status"]) == {"completed_manifested"}
    assert coverage.iloc[0]["status"] == "declared_roster_indexed"

    paths = write_research_trial_ledger(
        tmp_path / "reports",
        output_dir=tmp_path / "governance",
    )
    assert all(path.exists() for path in paths)
    assert "does not invent interrupted" in paths[-1].read_text(encoding="utf-8")


def test_trial_ledger_exposes_unmanifested_research_artifacts(tmp_path) -> None:
    orphan = tmp_path / "reports" / "abandoned_probe"
    orphan.mkdir(parents=True)
    (orphan / "summary.md").write_text("# Old probe\n", encoding="utf-8")

    ledger, coverage = build_research_trial_ledger(tmp_path / "reports")

    assert ledger.empty
    row = coverage.iloc[0]
    assert row["study"] == "abandoned_probe"
    assert row["status"] == "artifact_directory_without_manifest"
