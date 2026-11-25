from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import trade_bot.dashboard.trends as trends_module


def test_partial_materialized_history_still_checks_retained_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_metrics = _metric_history("materialized")

    class FakeWarehouse:
        def __init__(self, _path: str, *, read_only: bool = False) -> None:
            assert read_only

        def operating_history_frames(
            self,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            return stored_metrics, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    class EmptyRunStore:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            assert kwargs["read_only"] is True

        def list_snapshots(self, *, limit: int) -> pd.DataFrame:
            assert limit == 1_000
            return pd.DataFrame()

    monkeypatch.setattr(trends_module, "TradingWarehouse", FakeWarehouse)
    monkeypatch.setattr(trends_module, "RunStore", EmptyRunStore)

    metrics, components, drivers, rotation = trends_module._load_snapshot_trend_frames(
        "trade_bot.duckdb",
        "artifacts",
        "job_logs",
        limit=1_000,
        force_snapshot_reconstruction=False,
    )

    assert metrics["run_id"].tolist() == ["materialized"]
    assert components.empty
    assert drivers.empty
    assert rotation.empty


def test_snapshot_history_fallback_loads_artifacts_when_materialized_history_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_run_ids: list[str] = []

    class EmptyWarehouse:
        def __init__(self, _path: str, *, read_only: bool = False) -> None:
            assert read_only

        def operating_history_frames(
            self,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    class FakeRunStore:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            assert kwargs["read_only"] is True

        def list_snapshots(self, *, limit: int) -> pd.DataFrame:
            assert limit == 17
            return pd.DataFrame([{"run_id": "snapshot-run"}])

        def load_snapshot(self, run_id: str) -> tuple[SimpleNamespace, SimpleNamespace]:
            loaded_run_ids.append(run_id)
            return _snapshot_run(), SimpleNamespace(
                run_id=run_id,
                market_date="2026-07-20",
                created_at_utc="2026-07-21T00:19:14+00:00",
            )

    monkeypatch.setattr(trends_module, "TradingWarehouse", EmptyWarehouse)
    monkeypatch.setattr(trends_module, "RunStore", FakeRunStore)
    monkeypatch.setattr(
        trends_module,
        "_snapshot_driver_rotation_rows",
        lambda _run, _base: [],
    )

    metrics, _components, _drivers, _rotation = trends_module._load_snapshot_trend_frames(
        "trade_bot.duckdb",
        "artifacts",
        "job_logs",
        limit=17,
        force_snapshot_reconstruction=False,
    )

    assert loaded_run_ids == ["snapshot-run"]
    assert metrics["run_id"].tolist() == ["snapshot-run"]
    assert metrics.iloc[0]["risk_score"] == pytest.approx(0.42)


def test_forced_snapshot_reconstruction_combines_artifacts_with_materialized_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_metrics = _metric_history("materialized")

    class FakeWarehouse:
        def __init__(self, _path: str, *, read_only: bool = False) -> None:
            assert read_only

        def operating_history_frames(
            self,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            return stored_metrics, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    class FakeRunStore:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            assert kwargs["read_only"] is True

        def list_snapshots(self, *, limit: int) -> pd.DataFrame:
            assert limit == 3
            return pd.DataFrame([{"run_id": "snapshot-run"}])

        def load_snapshot(self, run_id: str) -> tuple[SimpleNamespace, SimpleNamespace]:
            return _snapshot_run(), SimpleNamespace(
                run_id=run_id,
                market_date="2026-07-20",
                created_at_utc="2026-07-21T00:19:14+00:00",
            )

    monkeypatch.setattr(trends_module, "TradingWarehouse", FakeWarehouse)
    monkeypatch.setattr(trends_module, "RunStore", FakeRunStore)
    monkeypatch.setattr(
        trends_module,
        "_snapshot_driver_rotation_rows",
        lambda _run, _base: [],
    )

    metrics, _components, _drivers, _rotation = trends_module._load_snapshot_trend_frames(
        "trade_bot.duckdb",
        "artifacts",
        "job_logs",
        limit=3,
        force_snapshot_reconstruction=True,
    )

    assert set(metrics["run_id"]) == {"materialized", "snapshot-run"}


def test_materialized_history_is_limited_to_retained_snapshot_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_metrics = pd.concat(
        [
            _metric_history("retained"),
            _metric_history("stale").assign(
                history_time="2026-07-19",
                market_date="2026-07-19",
            ),
        ],
        ignore_index=True,
    )

    class FakeWarehouse:
        def __init__(self, _path: str, *, read_only: bool = False) -> None:
            assert read_only

        def operating_history_frames(
            self,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            return stored_metrics, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    class FakeRunStore:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            assert kwargs["read_only"] is True

        def list_snapshots(self, *, limit: int) -> pd.DataFrame:
            assert limit == 3
            return pd.DataFrame(
                [{"run_id": "snapshot-run", "market_date": "2026-07-20"}]
            )

        def load_snapshot(self, run_id: str) -> tuple[SimpleNamespace, SimpleNamespace]:
            raise AssertionError(f"Materialized history should avoid loading {run_id}")

    monkeypatch.setattr(trends_module, "TradingWarehouse", FakeWarehouse)
    monkeypatch.setattr(trends_module, "RunStore", FakeRunStore)
    monkeypatch.setattr(trends_module, "_snapshot_driver_rotation_rows", lambda _run, _base: [])

    metrics, _components, _drivers, _rotation = trends_module._load_snapshot_trend_frames(
        "trade_bot.duckdb",
        "artifacts",
        "job_logs",
        limit=3,
        force_snapshot_reconstruction=False,
    )

    assert "stale" not in set(metrics["run_id"])


def test_compact_metric_line_figure_breaks_sparse_history_gaps() -> None:
    history = pd.DataFrame(
        {
            "history_time": ["2026-01-01", "2026-01-08", "2026-03-01"],
            "risk_score": [0.2, 0.3, 0.4],
        }
    )

    figure = trends_module.compact_metric_line_figure(
        history,
        columns=["risk_score"],
        title="Risk",
        break_gap_days=14,
    )

    assert list(figure.data[0].y) == [0.2, 0.3, None, 0.4]


def test_read_only_warehouse_skips_schema_setup_and_reads_existing_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "trade_bot.duckdb"
    writer = trends_module.TradingWarehouse(db_path)
    writer.save_operating_history(
        metrics=_metric_history("materialized").assign(history_id="materialized"),
        components=pd.DataFrame(),
        scenario_drivers=pd.DataFrame(),
        driver_rotation=pd.DataFrame(),
    )

    def unexpected_schema_setup(_warehouse: object) -> None:
        raise AssertionError("read-only warehouse must not run schema setup")

    monkeypatch.setattr(
        trends_module.TradingWarehouse,
        "_ensure_schema",
        unexpected_schema_setup,
    )
    reader = trends_module.TradingWarehouse(db_path, read_only=True)

    metrics, _components, _drivers, _rotation = reader.operating_history_frames()

    assert metrics["run_id"].tolist() == ["materialized"]


def test_strategy_history_filter_does_not_mix_promoted_strategies() -> None:
    metrics = pd.DataFrame(
        [
            {"strategy_name": "primary", "market_date": "2026-01-02"},
            {"strategy_name": "challenger", "market_date": "2026-01-02"},
        ]
    )
    empty = pd.DataFrame()

    filtered, _components, _drivers, _rotation = trends_module._filter_strategy_history(
        (metrics, empty, empty, empty),
        "challenger",
    )

    assert filtered["strategy_name"].tolist() == ["challenger"]


def _metric_history(run_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "history_time": "2026-07-20",
                "snapshot_time": "2026-07-21T00:19:14+00:00",
                "market_date": "2026-07-20",
                "run_id": run_id,
                "source": "test",
                "reconstruction_note": "test",
                "risk_score": 0.42,
            }
        ]
    )


def _snapshot_run() -> SimpleNamespace:
    return SimpleNamespace(
        current_state=SimpleNamespace(
            risk_score=0.42,
            regime_instability=pd.DataFrame(),
            regime_instability_components=pd.DataFrame(),
            scenario_drivers=pd.DataFrame(),
        ),
        trade_decision=SimpleNamespace(
            summary=pd.DataFrame(
                [
                    {
                        "one_month_risk_off_probability": 0.25,
                        "risk_budget_multiplier": 0.75,
                    }
                ]
            )
        ),
        portfolio_risk=SimpleNamespace(
            summary=pd.DataFrame(),
            correlation_regime=pd.DataFrame(),
        ),
        prices=pd.DataFrame(),
    )
