from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trade_bot.dashboard import forward_test as forward_test_module
from trade_bot.dashboard import loaders as dashboard_loaders
from trade_bot.dashboard_v2.pages import (
    command_center,
    full_workbench,
    monitoring,
    research,
    risk,
    simulation,
)
from trade_bot.dashboard_v2.services import runtime as runtime_service
from trade_bot.dashboard_v2.services import warehouse_service
from trade_bot.research.current_state import build_market_health, momentum_state_table


def test_v22_identity_and_readme_map_match_current_routes() -> None:
    app_source = Path("src/trade_bot/dashboard_v2/app.py").read_text()
    readme = Path("README.md").read_text()

    assert "Trade Bot V2.2" in app_source
    assert '"V2.2 workbench"' in app_source
    assert "Point-in-time snapshot mode pins the baseline run only" in app_source
    for route_label in [
        "Today",
        "Macro",
        "Risk",
        "Forward Test",
        "Monitoring",
        "Research",
        "Performance",
        "Launch",
        "Simulation",
    ]:
        assert f"| {route_label} |" in readme


def test_research_frontier_defers_candidate_backtest_diagnostics() -> None:
    source = inspect.getsource(research.render_research_page)

    assert "include_candidate_diagnostics=False" in source


def test_simulation_summary_uses_persisted_median_abs_error_column() -> None:
    source = inspect.getsource(simulation.render_simulation_page)

    assert '"primary_median_abs_error"' in source
    assert "primary_median_absolute_error" not in source


def test_simulation_separates_validation_context_and_action_readiness() -> None:
    runs = pd.DataFrame(
        [
            {
                "validation_run_id": "validation-1",
                "strategy": "validated_strategy",
                "market_date": "2026-06-30",
                "snapshot_run_id": "snapshot-old",
            }
        ]
    )
    weak_action_metrics = pd.DataFrame(
        [
            {
                "metric_scope": "primary_summary",
                "launch_decision_accuracy": 0.26,
                "launch_action_score": 0.57,
                "launch_overrisk_rate": 0.42,
            }
        ]
    )
    strong_action_metrics = weak_action_metrics.assign(
        launch_decision_accuracy=0.65,
        launch_action_score=0.80,
        launch_overrisk_rate=0.20,
    )

    context = simulation._validation_context_read(
        runs,
        current_market_date="2026-07-20",
        promoted_strategy="promoted_strategy",
    )

    assert "validated_strategy" in context
    assert "2026-06-30" in context
    assert "2026-07-20" in context
    assert "promoted_strategy" in context
    assert "applies only to the named strategy and run" in context
    assert simulation._decision_readiness(weak_action_metrics) == "Not decision-ready"
    assert simulation._decision_readiness(strong_action_metrics) == "Action checks passed"
    assert simulation._decision_readiness(pd.DataFrame()) == "Not evaluated"


def test_simulation_prefers_separate_persisted_calibration_and_action_reads() -> None:
    runs = pd.DataFrame(
        [
            {
                "primary_distribution_calibration_read": ("return_bands_calibrated_for_research"),
                "primary_action_readiness_read": "action_checks_not_ready",
                "primary_validity_read": "return_bands_and_action_checks_ready_for_research",
            }
        ]
    )

    assert simulation._distribution_calibration_read(runs, pd.DataFrame()) == (
        "return_bands_calibrated_for_research"
    )
    assert simulation._action_readiness_read(runs, pd.DataFrame()) == ("action_checks_not_ready")

    legacy = pd.DataFrame(
        [{"primary_validity_read": "return_bands_calibrated__action_checks_marginal"}]
    )
    assert simulation._distribution_calibration_read(legacy, pd.DataFrame()) == (
        "return_bands_calibrated_for_research"
    )
    assert simulation._action_readiness_read(legacy, pd.DataFrame()) == ("action_checks_marginal")


def test_missing_read_only_dashboard_stores_return_empty_without_creating_db(tmp_path) -> None:
    store_path = tmp_path / "missing.duckdb"
    artifact_dir = tmp_path / "artifacts"
    job_log_dir = tmp_path / "jobs"
    missing_config = tmp_path / "missing.yaml"

    assert (
        dashboard_loaders.load_snapshot_dashboard_run(
            str(missing_config),
            str(missing_config),
            str(missing_config),
            str(missing_config),
            str(store_path),
            str(artifact_dir),
            str(job_log_dir),
        )
        is None
    )
    assert (
        dashboard_loaders.load_snapshot_dashboard_run_by_id(
            str(store_path),
            str(artifact_dir),
            str(job_log_dir),
            "missing-run",
        )
        is None
    )
    assert (
        dashboard_loaders.load_previous_snapshot_dashboard_run(
            str(missing_config),
            str(missing_config),
            str(missing_config),
            str(missing_config),
            str(store_path),
            str(artifact_dir),
            str(job_log_dir),
        )
        is None
    )
    assert dashboard_loaders.load_snapshot_jobs_frame(
        str(store_path), str(artifact_dir), str(job_log_dir)
    ).empty

    paths = runtime_service.DashboardPaths(
        run_store_path=store_path,
        artifact_dir=artifact_dir,
        job_log_dir=job_log_dir,
    )
    assert runtime_service.snapshot_choices(paths).empty
    assert warehouse_service.read_warehouse_table(store_path, "run_snapshots").empty
    assert warehouse_service.champion_challenger_frame(store_path).empty
    assert warehouse_service.monitoring_windows(store_path).empty
    assert warehouse_service.warehouse_counts(store_path).empty
    runs, metrics = warehouse_service.simulation_validation_summary(store_path)
    assert runs.empty
    assert metrics.empty
    assert not store_path.exists()


def test_selected_snapshot_runtime_suppresses_current_operating_claims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline_run = SimpleNamespace(trade_decision=object())
    manifest = SimpleNamespace(run_id="historical-run")
    promoted_book = SimpleNamespace(
        book_id="book-1",
        book_name="Promoted",
        mode="paper",
        account="ira",
        strategy_name="strategy_a",
        account_value=100_000.0,
    )

    class FakeJournal:
        def __init__(self, _path: Path) -> None:
            pass

        def get_promoted_book(self) -> SimpleNamespace:
            return promoted_book

        def load_recommendation_tickets(self, **_kwargs: object) -> pd.DataFrame:
            raise AssertionError("historical mode must not query current tickets")

    monkeypatch.setattr(runtime_service, "load_config", lambda _path: object())
    monkeypatch.setattr(
        runtime_service,
        "load_snapshot_dashboard_run_by_id",
        lambda *_args: (baseline_run, manifest),
    )
    monkeypatch.setattr(runtime_service, "TradeJournal", FakeJournal)
    monkeypatch.setattr(
        runtime_service,
        "resolve_trade_decision_for_strategy",
        lambda *_args, **_kwargs: pytest.fail(
            "historical mode must not resolve a current operating decision"
        ),
    )

    runtime = runtime_service.load_runtime(
        paths=runtime_service.DashboardPaths(
            config_path=tmp_path / "config.yaml",
            journal_path=tmp_path / "journal.sqlite",
            run_store_path=tmp_path / "store.duckdb",
        ),
        run_source="Selected snapshot",
        selected_snapshot_run_id="historical-run",
    )

    assert runtime.is_historical_snapshot_mode
    assert runtime.snapshot_loaded
    assert runtime.operating_trade_decision is None
    assert runtime.open_ticket_count is None
    assert runtime.book_alignment is None
    assert runtime.execution_book_alignment is None
    assert runtime.action_headline is None
    assert runtime.operating_strategy_error == runtime_service.HISTORICAL_SNAPSHOT_NOTICE

    assert "is_historical_snapshot_mode" in inspect.getsource(command_center.render_today_page)
    assert "is_historical_snapshot_mode" in inspect.getsource(risk.render_risk_page)
    forward_source = inspect.getsource(full_workbench.render_forward_test_page)
    assert "allow_decision_actions=not runtime.is_historical_snapshot_mode" in forward_source


def test_historical_forward_test_keeps_book_selector_but_stops_before_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[str] = []
    fake_streamlit = SimpleNamespace(
        subheader=lambda _value: rendered.append("heading"),
        caption=lambda _value: rendered.append("caption"),
        warning=lambda _value: rendered.append("warning"),
    )
    selected_book = SimpleNamespace(book_id="book-1")

    def select_book() -> SimpleNamespace:
        rendered.append("book-selector")
        return selected_book

    monkeypatch.setattr(forward_test_module, "st", fake_streamlit)
    monkeypatch.setattr(
        forward_test_module,
        "resolve_trade_decision_for_strategy",
        lambda *_args, **_kwargs: pytest.fail(
            "historical mode must stop before resolving a trade decision"
        ),
    )

    forward_test_module._render_forward_test_and_journal(
        SimpleNamespace(),
        SimpleNamespace(),
        selected_book=selected_book,
        book_selector=select_book,
        allow_decision_actions=False,
        decision_actions_disabled_reason="Historical display only",
    )

    assert rendered.index("book-selector") < rendered.index("warning")
    assert rendered[-1] == "caption"


def test_monitoring_current_valuation_requires_snapshot_date_match() -> None:
    frame = pd.DataFrame(
        {
            "window_id": ["old", "current", "missing"],
            "valuation_date": ["2026-07-19", "2026-07-20T18:00:00Z", None],
            "forward_status": [
                "ahead_of_benchmark",
                "lagging_benchmark",
                "ahead_of_benchmark",
            ],
        }
    )

    valued = monitoring._valuation_rows_for_market_date(frame, "2026-07-20")

    assert valued["window_id"].tolist() == ["current"]
    assert monitoring._rows_with_forward_status(valued, "ahead_of_benchmark").empty
    assert monitoring._rows_with_forward_status(valued, "lagging_benchmark")[
        "window_id"
    ].tolist() == ["current"]
    assert "Latest paper valuation: 2026-07-20" in monitoring._monitoring_freshness_read(
        frame, "2026-07-21"
    )


def test_monitoring_default_trends_preserve_champion_and_reference_visibility() -> None:
    rows: list[dict[str, object]] = []
    for index in range(10):
        rows.append(
            {
                "history_time": "2026-07-20",
                "window_label": f"challenger-{index}",
                "window_role": "challenger",
                "cumulative_return": index / 100.0,
            }
        )
    rows.extend(
        [
            {
                "history_time": "2026-07-20",
                "window_label": "champion-low-return",
                "window_role": "champion",
                "cumulative_return": 0.001,
            },
            {
                "history_time": "2026-07-20",
                "window_label": "reference-low-return",
                "window_role": "reference",
                "cumulative_return": 0.0,
            },
        ]
    )
    labeled = monitoring._with_monitoring_trend_labels(pd.DataFrame(rows))

    defaults = monitoring._default_monitoring_trend_labels(labeled, limit=8)

    assert "Champion | champion-low-return" in defaults
    assert "Reference | reference-low-return" in defaults
    assert len(defaults) == 8


def test_risk_uses_promoted_trade_decision_and_sanitizes_old_vixy_snapshots() -> None:
    operating_risk = object()
    baseline_risk = object()
    runtime = SimpleNamespace(
        operating_trade_decision=SimpleNamespace(portfolio_risk=operating_risk),
        baseline_run=SimpleNamespace(
            portfolio_risk=baseline_risk,
            trade_decision=SimpleNamespace(portfolio_risk=baseline_risk),
        ),
    )
    legacy_health = pd.DataFrame(
        {
            "return_1m": [0.01, 0.20],
            "drawdown": [-0.05, -0.9999],
        },
        index=["SPY", "VIXY"],
    )

    display = risk._market_health_for_display(legacy_health)

    assert risk._portfolio_risk(runtime) is operating_risk
    assert pd.isna(display.loc["VIXY", "drawdown"])
    assert display.loc["SPY", "drawdown"] == -0.05
    assert display.loc["VIXY", "drawdown_basis"] == "not_applicable_short_term_volatility_proxy"


def test_research_provenance_warning_flags_integrity_and_stale_source() -> None:
    current = pd.DataFrame(
        [
            {
                "artifact_integrity_status": "verified",
                "missing_artifact_count": 0,
                "artifact_mismatch_count": 0,
                "source_tree_status": "current",
            }
        ]
    )
    invalid = pd.DataFrame(
        [
            {
                "artifact_integrity_status": "hash_or_size_mismatch",
                "missing_artifact_count": 1,
                "artifact_mismatch_count": 2,
                "source_tree_status": "stale",
            }
        ]
    )

    assert research._provenance_integrity_warning(current) is None
    warning = research._provenance_integrity_warning(invalid)
    assert warning is not None
    assert "1 declared artifact(s) missing" in warning
    assert "2 artifact hash/size mismatch(es)" in warning
    assert "different source tree" in warning


def test_market_health_does_not_treat_vixy_lifetime_decay_as_current_drawdown() -> None:
    index = pd.bdate_range("2024-01-01", periods=300)
    prices = pd.DataFrame(
        {
            "SPY": np.linspace(100.0, 150.0, len(index)),
            "VIXY": np.linspace(100.0, 1.0, len(index)),
        },
        index=index,
    )
    momentum = momentum_state_table(prices)

    health = build_market_health(prices, momentum)

    assert pd.isna(health.loc["VIXY", "drawdown"])
    assert health.loc["VIXY", "drawdown_basis"] == "not_applicable_short_term_volatility_proxy"
    assert health.loc["VIXY", "return_1m"] < 0.0
    assert health.loc["VIXY", "return_3m"] < 0.0
    assert health.loc["SPY", "drawdown"] == 0.0
