from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from trade_bot.research.strategy_source_audit import build_strategy_source_audit


def test_strategy_source_audit_separates_snapshot_and_experiment_scorecards(
    tmp_path: Path,
) -> None:
    warehouse_path = tmp_path / "trade_bot.duckdb"
    snapshot_metrics = pd.DataFrame(
        {
            "run_id": ["snapshot_1", "snapshot_1"],
            "market_date": ["2026-07-20", "2026-07-20"],
            "updated_at_utc": ["2026-07-20T20:00:00+00:00", "2026-07-20T20:00:00+00:00"],
            "strategy": ["runtime_22pct", "runtime_reference"],
            "final_equity": [7_000_000.0, 900_000.0],
            "cagr": [0.2197, 0.109],
            "max_drawdown": [-0.203, -0.55],
            "calmar": [1.08, 0.20],
            "sharpe": [1.2, 0.6],
        }
    )
    with duckdb.connect(str(warehouse_path)) as connection:
        connection.register("snapshot_metrics", snapshot_metrics)
        connection.execute("CREATE TABLE snapshot_strategy_metrics AS SELECT * FROM snapshot_metrics")

    experiment_root = tmp_path / "experiments"
    iteration_dir = experiment_root / "iteration_111"
    iteration_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "strategy": ["experiment_16pct"],
            "cagr": [0.165],
            "max_drawdown": [-0.239],
            "calmar": [0.69],
            "promotion_score": [0.54],
        }
    ).to_csv(iteration_dir / "scorecard.csv", index=False)

    rolling_root = tmp_path / "reports"
    rolling_root.mkdir()
    pd.DataFrame(
        {
            "strategy": ["rolling_22pct"],
            "cagr": [0.221],
            "cagr_win_rate": [0.99],
            "delta_cagr": [0.20],
            "max_drawdown": [-0.10],
        }
    ).to_csv(rolling_root / "rolling_deltas.csv", index=False)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "note.md").write_text(
        "Runtime snapshot candidates clustered near the 20-22 percent historical CAGR area.\n",
        encoding="utf-8",
    )

    audit = build_strategy_source_audit(
        warehouse_path=warehouse_path,
        experiment_roots=(experiment_root,),
        text_roots=(docs_root,),
        scan_roots=(rolling_root,),
        top_n=10,
    )

    top = audit.full_history_top.iloc[0]
    assert top["strategy"] == "runtime_22pct"
    assert top["source_scope"] == "runtime_snapshot_full_history"
    assert "experiment_scorecard_full_history" in set(audit.full_history_top["source_scope"])

    hits = audit.high_cagr_metric_hits
    assert "runtime_snapshot_full_history" in set(hits["source_scope"])
    assert "rolling_window_diagnostic" in set(hits["source_scope"])
    assert "cagr_win_rate" not in set(hits["metric_column"])
    assert "delta_cagr" not in set(hits["metric_column"])

    assert audit.ambiguous_references.loc[0, "reference_scope"] == "runtime_snapshot_reference"
    assert "runtime_snapshot_full_history" in audit.summary
