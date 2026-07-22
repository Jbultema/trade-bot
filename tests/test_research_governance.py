from __future__ import annotations

import json

import pandas as pd

from trade_bot.research.research_governance import (
    audit_point_in_time_universe,
    build_research_trial_ledger,
    write_research_trial_ledger,
)


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
