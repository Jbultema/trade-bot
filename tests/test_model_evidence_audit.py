from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from trade_bot.config import load_config
from trade_bot.research.model_evidence_audit import (
    build_model_evidence_audit,
    build_relevant_model_slate,
    write_model_evidence_audit,
)


def _scorecards(config_path: str = "configs/baseline.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    primary = config.primary_strategy
    return pd.DataFrame(
        [
            {
                "strategy": primary,
                "iteration": 1,
                "iteration_rank": 1,
                "cagr": 0.20,
                "max_drawdown": -0.25,
                "growth_constrained_utility_score": 0.9,
                "is_growth_pareto_efficient": True,
                "strategy_json": config.strategies[primary].model_dump_json(),
                "walk_forward_positive_rate": 0.7,
                "worst_regime_return": -0.1,
                "operability_score": 0.8,
                "transition_regime_hit_rate": 0.6,
                "oos_evidence_status": "missing_oos_evidence",
            },
            {
                "strategy": "buy_hold_qqq",
                "iteration": 1,
                "iteration_rank": 2,
                "cagr": 0.15,
                "max_drawdown": -0.53,
                "growth_constrained_utility_score": 0.4,
                "is_growth_pareto_efficient": False,
                "strategy_json": config.strategies["buy_hold_qqq"].model_dump_json(),
            },
        ]
    )


def test_relevant_slate_contains_configured_primary_and_reference() -> None:
    config = load_config("configs/baseline.yaml")
    slate = build_relevant_model_slate(config, _scorecards(), top_n=1)

    assert config.primary_strategy in set(slate["strategy"])
    assert set(config.strategies).issubset(set(slate["strategy"]))
    primary = slate.loc[slate["strategy"].eq(config.primary_strategy)].iloc[0]
    qqq = slate.loc[slate["strategy"].eq("buy_hold_qqq")].iloc[0]
    assert bool(primary["primary"]) is True
    assert bool(qqq["reference"]) is True


def test_audit_does_not_mislabel_walk_forward_as_selection_oos(tmp_path: Path) -> None:
    config = load_config("configs/baseline.yaml")
    experiment_root = tmp_path / "experiments"
    iteration = experiment_root / "iteration_01"
    iteration.mkdir(parents=True)
    for filename in (
        "metrics.csv",
        "window_summary.csv",
        "walk_forward_summary.csv",
        "regime_metrics.csv",
        "operability_metrics.csv",
        "transition_metrics.csv",
    ):
        (iteration / filename).write_text("strategy,value\nexample,1\n", encoding="utf-8")

    audit = build_model_evidence_audit(
        config,
        _scorecards(),
        experiment_root=experiment_root,
        report_root=tmp_path / "reports",
        selection_cutoff=date(2026, 7, 24),
        top_n=1,
    )
    primary = audit.evidence_matrix.loc[
        audit.evidence_matrix["strategy"].eq(config.primary_strategy)
    ].iloc[0]

    assert primary["temporal_walk_forward"] == "pass"
    assert primary["selection_oos_status"] == "missing"
    assert primary["prospective_shadow_status"] == "frozen_pending"


def test_audit_persists_freeze_registry_and_manifest(tmp_path: Path) -> None:
    config = load_config("configs/baseline.yaml")
    audit = build_model_evidence_audit(
        config,
        _scorecards(),
        experiment_root=tmp_path / "missing",
        report_root=tmp_path / "reports",
        selection_cutoff=date(2026, 7, 24),
        top_n=1,
    )
    artifacts = write_model_evidence_audit(audit, tmp_path / "audit")
    registry = pd.read_csv(artifacts["prospective"])

    assert artifacts["manifest"].is_file()
    assert set(registry["status"]) == {"frozen_pending"}
    assert set(registry["selection_cutoff"]) == {"2026-07-24"}
    assert set(registry["minimum_observations"]) == {252}
    assert registry["definition_sha256"].str.len().eq(64).all()
