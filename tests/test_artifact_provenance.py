from __future__ import annotations

import json

import numpy as np
import pandas as pd

from trade_bot.config import ExecutionConfig
from trade_bot.research.artifact_provenance import (
    build_runtime_provenance,
    verify_research_manifest,
    write_research_manifest,
)


def test_research_manifest_captures_config_code_and_price_identity(tmp_path) -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=dates)
    (tmp_path / "summary.md").write_text("summary\n", encoding="utf-8")
    (tmp_path / "metrics.csv").write_text("metric,value\ncagr,0.1\n", encoding="utf-8")

    path = write_research_manifest(
        tmp_path,
        study="unit_test",
        config=ExecutionConfig(transaction_cost_bps=7.5),
        prices=prices,
        parameters={
            "candidate_set": ["raw", "ewm5"],
            "numpy_values": np.array([1.0, np.nan]),
            "missing": pd.NA,
        },
        artifacts=["summary.md", "metrics.csv"],
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["study"] == "unit_test"
    assert manifest["schema_version"] == 4
    assert manifest["automatic_promotion_allowed"] is False
    assert manifest["config"]["transaction_cost_bps"] == 7.5
    assert manifest["price_input"]["market_date"] == "2024-01-03"
    assert manifest["price_input"]["frame_sha256"]
    assert manifest["code"]["source_tree_sha256"]
    assert manifest["code"]["git_tree_sha"]
    assert manifest["code"]["poetry_lock_sha256"]
    assert manifest["code"]["pyproject_sha256"]
    assert manifest["artifacts"] == ["metrics.csv", "summary.md"]
    integrity = {row["path"]: row for row in manifest["artifact_integrity"]}
    assert integrity["summary.md"]["size_bytes"] == len(b"summary\n")
    assert len(integrity["summary.md"]["sha256"]) == 64
    assert integrity["metrics.csv"]["size_bytes"] == len(b"metric,value\ncagr,0.1\n")
    assert manifest["parameters"]["numpy_values"] == [1.0, None]
    assert manifest["parameters"]["missing"] is None
    assert (
        manifest["research_governance"]["point_in_time_universe"]["status"]
        == "missing_point_in_time_membership"
    )
    assert (
        manifest["research_governance"]["promotion_evidence_gate"]
        == "blocked_missing_or_invalid_oos_evidence"
    )
    oos = manifest["research_governance"]["out_of_sample_evidence"]
    assert oos["status"] == "missing_oos_evidence"
    assert oos["promotion_eligible"] is False

    verification = verify_research_manifest(
        path,
        current_source_tree_sha256=manifest["code"]["source_tree_sha256"],
    )
    assert verification["artifact_integrity_status"] == "verified"
    assert verification["verified_artifact_count"] == 2
    assert verification["source_tree_status"] == "current"


def test_research_manifest_verification_flags_tampering_and_missing_artifacts(tmp_path) -> None:
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("metric,value\ncagr,0.1\n", encoding="utf-8")
    manifest_path = write_research_manifest(
        tmp_path,
        study="tamper_test",
        config=ExecutionConfig(),
        artifacts=["metrics.csv"],
    )

    artifact.write_text("metric,value\ncagr,0.9\n", encoding="utf-8")
    mismatch = verify_research_manifest(
        manifest_path,
        current_source_tree_sha256="different-source-tree",
    )
    assert mismatch["artifact_integrity_status"] == "hash_or_size_mismatch"
    assert mismatch["artifact_mismatch_count"] == 1
    assert mismatch["source_tree_status"] == "stale"

    artifact.unlink()
    missing = verify_research_manifest(manifest_path)
    assert missing["artifact_integrity_status"] == "missing_artifacts"
    assert missing["missing_artifact_count"] == 1


def test_manifest_only_allows_human_review_with_verified_oos_and_universe(
    tmp_path,
) -> None:
    dates = pd.bdate_range("2025-01-02", periods=5)
    prices = pd.DataFrame({"SPY": np.arange(100.0, 105.0)}, index=dates)
    membership = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "effective_from": ["2000-01-01"],
            "effective_to": [None],
            "source": ["fund_inception_record"],
            "source_as_of": ["2026-07-24"],
            "delisting_return_included": [True],
            "delisting_return_source": ["total_return_vendor"],
        }
    )

    path = write_research_manifest(
        tmp_path,
        study="sealed_holdout_test",
        config=ExecutionConfig(),
        prices=prices,
        universe_membership=membership,
        oos_evidence={
            "method": "sealed_holdout",
            "completed": True,
            "parameters_frozen_before_evaluation": True,
            "evaluation_used_for_model_selection": False,
            "primary_metric_pre_registered": True,
            "selection_cutoff": "2024-12-31",
            "evaluation_start": "2025-01-02",
            "evaluation_end": "2025-01-08",
            "observation_count": 5,
            "minimum_observations": 5,
        },
        parameters={"candidate_set": ["frozen_candidate"]},
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    governance = manifest["research_governance"]
    assert governance["out_of_sample_evidence"]["status"] == "verified_sealed_holdout"
    assert governance["promotion_evidence_gate"] == "eligible_for_human_review"
    assert manifest["research_status"] == "oos_evaluated_human_review_required"
    assert manifest["automatic_promotion_allowed"] is False


def test_source_tree_identity_ignores_python_cache_but_tracks_source(tmp_path) -> None:
    source = tmp_path / "src" / "trade_bot" / "research" / "study.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "baseline.yaml").write_text("mode: test\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "poetry.lock").write_text("", encoding="utf-8")

    initial = build_runtime_provenance(repo_root=tmp_path)["source_tree_sha256"]
    cache = source.parent / "__pycache__" / "study.cpython-312.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"first cache")
    after_cache = build_runtime_provenance(repo_root=tmp_path)["source_tree_sha256"]
    cache.write_bytes(b"rewritten cache")
    after_cache_rewrite = build_runtime_provenance(repo_root=tmp_path)["source_tree_sha256"]

    assert after_cache == initial
    assert after_cache_rewrite == initial

    source.write_text("VALUE = 2\n", encoding="utf-8")
    after_source_change = build_runtime_provenance(repo_root=tmp_path)["source_tree_sha256"]
    assert after_source_change != initial
