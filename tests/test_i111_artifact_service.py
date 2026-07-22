from __future__ import annotations

import pandas as pd

from trade_bot.config import load_config
from trade_bot.dashboard_v2.services.artifact_service import i111_evidence_frames
from trade_bot.DEFAULTS import DEFAULT_CONFIG_PATH
from trade_bot.research.artifact_provenance import write_research_manifest


def test_i111_evidence_frames_include_metrics_and_provenance(tmp_path) -> None:
    smoothing = tmp_path / "i111_execution_smoothing"
    smoothing.mkdir()
    pd.DataFrame(
        [
            {
                "transform": "ewm5",
                "local_pbo": True,
                "retrospective_gate_pass": True,
            }
        ]
    ).to_csv(
        smoothing / "promotion_gates.csv",
        index=False,
    )
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0]},
        index=pd.to_datetime(["2026-07-17", "2026-07-20"]),
    )
    write_research_manifest(
        smoothing,
        study="i111_execution_smoothing_v2_3",
        config=load_config(DEFAULT_CONFIG_PATH),
        prices=prices,
        artifacts=["promotion_gates.csv"],
    )

    frames = i111_evidence_frames(tmp_path)

    assert frames["smoothing_gates"].iloc[0]["transform"] == "ewm5"
    assert bool(frames["smoothing_gates"].iloc[0]["family_pbo_gate"])
    assert "local_pbo" not in frames["smoothing_gates"]
    assert frames["smoothing_gates"].iloc[0]["pbo_scope"] == "family_15_strategies_70_splits"
    assert frames["manifests"].iloc[0]["study"] == "i111_execution_smoothing_v2_3"
    assert frames["manifests"].iloc[0]["market_date"] == "2026-07-20"
    assert not bool(frames["manifests"].iloc[0]["automatic_promotion_allowed"])
    assert frames["manifests"].iloc[0]["artifact_integrity_status"] == "verified"
    assert frames["manifests"].iloc[0]["source_tree_status"] == "current"
