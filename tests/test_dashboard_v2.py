from __future__ import annotations

from datetime import date
import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go

from trade_bot import DEFAULTS
from trade_bot.dashboard import forward_test, launch_lab
from trade_bot.dashboard.trends import filter_history_time_range
from trade_bot.dashboard_v2 import routes
from trade_bot.dashboard_v2.components import cards, tones
from trade_bot.dashboard_v2.pages import (
    command_center,
    full_workbench,
    macro,
    monitoring,
    research,
    risk,
    simulation,
)
from trade_bot.dashboard_v2.services.artifact_service import pbo_frames, read_csv_artifact
from trade_bot.dashboard_v2.services import runtime as runtime_service


def test_dashboard_v2_routes_cover_core_workflows() -> None:
    route_list = routes.routes()
    labels = [route.label for route in route_list]

    assert labels == [
        "Today",
        "Macro",
        "Risk",
        "Forward Test",
        "Monitoring",
        "Research",
        "Performance",
        "Launch",
        "Simulation",
    ]
    assert {route.lane for route in route_list} == {"Operate", "Decide", "Research"}
    assert routes.route_by_key("missing").key == "today"
    assert routes.route_by_key("research").runtime == "Fast by default"
    assert routes.route_by_key("simulation").runtime == "Fast by default"
    assert routes.route_by_key("macro").runtime == "Fast by default"


def test_dashboard_v2_uses_named_book_selector_for_runtime_and_forward_test() -> None:
    app_source = Path("src/trade_bot/dashboard_v2/app.py").read_text()
    selector_source = inspect.getsource(runtime_service.render_book_selector)
    runtime_source = inspect.getsource(runtime_service.load_runtime)
    forward_page_source = inspect.getsource(full_workbench.render_forward_test_page)
    forward_source = inspect.getsource(forward_test._render_forward_test_and_journal)

    assert "render_book_selector(paths.journal_path)" not in app_source
    assert "selected_book=book_selection.selected_book" not in app_source
    assert "promoted_book=book_selection.promoted_book" not in app_source
    assert "Create named book" in selector_source
    assert "Book controls" in selector_source
    assert "delete_book" in selector_source
    assert "Strategy to follow" in selector_source
    assert "_strategy_selector_options" in selector_source
    assert "mode=promoted_book.mode" in runtime_source
    assert "account=promoted_book.account" in runtime_source
    assert "operating_trade_decision" in runtime_source
    assert "selected_book=runtime.selected_book" in forward_page_source
    assert "baseline_run=runtime.baseline_run" in forward_page_source
    assert "bot_config=runtime.bot_config" in forward_page_source
    assert "book_selector()" in forward_source
    assert "resolve_trade_decision_for_strategy" in forward_source
    assert forward_source.index("Forward Test / Trade Journal") < forward_source.index(
        "book_selector()"
    )
    assert forward_source.index("book_selector()") < forward_source.index("brief-grid")


def test_dashboard_v2_book_strategy_options_prioritize_primary_and_preserve_existing() -> None:
    baseline_run = SimpleNamespace(
        results={
            "primary_candidate": object(),
            "challenger_candidate": object(),
        }
    )
    bot_config = SimpleNamespace(primary_strategy="primary_candidate")

    options = runtime_service._strategy_selector_options(
        current_values=["journal_only_scope"],
        baseline_run=baseline_run,
        bot_config=bot_config,
    )

    assert options["strategy_name"].tolist() == [
        DEFAULTS.DEFAULT_FORWARD_TEST_STRATEGY,
        "primary_candidate",
        "challenger_candidate",
        "journal_only_scope",
    ]
    assert "Scenario-adjusted trade decision" in str(options.iloc[0]["label"])
    assert "Existing journal value" in str(options.iloc[-1]["label"])


def test_dashboard_v2_native_pages_are_summary_first() -> None:
    today_source = inspect.getsource(command_center.render_today_page)
    monitoring_source = inspect.getsource(monitoring.render_monitoring_page)
    research_source = inspect.getsource(research.render_research_page)
    simulation_source = inspect.getsource(simulation.render_simulation_page)
    macro_source = inspect.getsource(macro.render_macro_page)
    risk_source = inspect.getsource(risk.render_risk_page)

    assert "st.tabs(" not in today_source
    assert "st.tabs(" not in monitoring_source
    assert "st.tabs(" not in research_source
    assert "st.tabs(" not in simulation_source
    assert "st.tabs(" not in macro_source
    assert "st.tabs(" not in risk_source
    assert today_source.index("Decision") < today_source.index("load_snapshot_trend_frames")
    assert monitoring_source.index("Trends") < monitoring_source.index("Readout")
    assert monitoring_source.index("Readout") < monitoring_source.index("Controls")
    assert monitoring_source.index("Controls") < monitoring_source.index("Full Workbench")
    assert monitoring_source.index("Trends") < monitoring_source.index("_render_monitoring(")
    assert research_source.index("Outcome Frontier") < research_source.index("_render_research_lab(")
    assert simulation_source.index('"Strategy simulations"') < simulation_source.index('"Validation"')
    assert simulation_source.index('"Validation"') < simulation_source.index('"Full Workbench"')
    assert simulation_source.index("_render_simulation_lab_direct_view") < simulation_source.index(
        "_render_simulation_lab("
    )
    assert "Full legacy workbench" not in simulation_source
    assert macro_source.index("Overview") < macro_source.index("Visual Explorer")
    assert macro_source.index("Visual Explorer") < macro_source.index("Signal Drivers")
    assert macro_source.index("Signal Drivers") < macro_source.index("News & Events")
    assert macro_source.index("News & Events") < macro_source.index("Full Workbench")
    assert macro_source.index("Overview") < macro_source.index("_render_news_and_macro(")
    risk_view_start = risk_source.index('"Risk view"')
    assert risk_source.index('"Overview"', risk_view_start) < risk_source.index(
        '"Portfolio Risk"', risk_view_start
    )
    assert risk_source.index('"Portfolio Risk"', risk_view_start) < risk_source.index(
        '"Operating Exposure"', risk_view_start
    )
    assert risk_source.index('"Operating Exposure"', risk_view_start) < risk_source.index(
        '"Instability"', risk_view_start
    )
    assert risk_source.index('"Instability"', risk_view_start) < risk_source.index(
        '"Scenarios"', risk_view_start
    )
    assert risk_source.index('"Scenarios"', risk_view_start) < risk_source.index(
        '"Confirmation"', risk_view_start
    )
    assert risk_source.index('"Confirmation"', risk_view_start) < risk_source.index(
        '"Momentum"', risk_view_start
    )
    assert risk_source.index('"Momentum"', risk_view_start) < risk_source.index(
        '"Full Workbench"', risk_view_start
    )
    assert risk_source.index("Overview") < risk_source.index("_render_risk_and_scenarios(")


def test_dashboard_v2_monitoring_trends_have_time_window_controls() -> None:
    monitoring_source = inspect.getsource(monitoring.render_monitoring_page)
    legacy_source = inspect.getsource(monitoring._render_monitoring_trend_range_controls)

    assert "_render_monitoring_trend_range_controls(trends)" in monitoring_source
    assert "Time range" in legacy_source
    assert "dashboard_v2_monitoring_trend_range" in legacy_source
    assert "filter_history_time_range(" in legacy_source


def test_filter_history_time_range_supports_presets_and_custom_dates() -> None:
    frame = pd.DataFrame(
        {
            "history_time": pd.to_datetime(
                ["2026-01-01", "2026-03-15", "2026-06-15", "2026-07-15"]
            ),
            "value": [1, 2, 3, 4],
        }
    )

    one_month = filter_history_time_range(frame, "1M")
    assert one_month["value"].tolist() == [3, 4]

    ytd = filter_history_time_range(frame, "YTD")
    assert ytd["value"].tolist() == [1, 2, 3, 4]

    custom = filter_history_time_range(
        frame,
        "Custom",
        custom_start=pd.Timestamp("2026-03-01"),
        custom_end=pd.Timestamp("2026-06-30"),
    )
    assert custom["value"].tolist() == [2, 3]


def test_macro_history_presets_snap_to_available_market_dates() -> None:
    prices = pd.DataFrame(
        {"SPY": range(190)},
        index=pd.bdate_range("2020-01-01", periods=190),
    )

    options = macro._macro_history_preset_options(prices)
    snapped = macro._market_date_on_or_before(prices, date(2020, 3, 22))
    sliced = macro._prices_as_of(prices, pd.Timestamp("2020-03-20"))

    assert options[0] == "Current snapshot"
    assert options[-1] == "Custom date"
    assert any("COVID liquidity crash - unwind" in option for option in options)
    assert not any("Global Financial Crisis" in option for option in options)
    assert snapped is not None
    assert snapped.date().isoformat() == "2020-03-20"
    assert sliced.index.max().date().isoformat() == "2020-03-20"


def test_macro_driver_figures_retitle_historical_context() -> None:
    rotation = pd.DataFrame(
        [
            {
                "driver_label": "Credit conditions",
                "model_role": "allocation_driver",
                "primary_rotation_state": "normally_important_active",
                "proven_relevance": 0.8,
                "current_activation": 0.7,
                "previous_30d_activation": 0.3,
                "previous_90d_activation": 0.2,
                "change_30d": 0.4,
                "change_90d": 0.5,
                "data_support": "market_price_proxy",
            }
        ]
    )
    context = macro._MacroDriverContext(
        as_of_date=pd.Timestamp("2020-03-20"),
        current_state=SimpleNamespace(),
        driver_rotation=rotation,
        driver_summary={"answer": "1 active driver"},
        narrative_summary={"plain": "none"},
        is_current=False,
        requested_label="COVID liquidity crash - unwind",
    )

    scatter, heatmap = macro._driver_rotation_figures_for_context(context)

    assert "as of 2020-03-20" in scatter.layout.title.text
    assert heatmap.data[0].x[2] == "Selected-date activation"


def test_dashboard_v2_is_primary_entrypoint_with_archived_v1_fallback() -> None:
    assert DEFAULTS.DEFAULT_DASHBOARD_APP_PATH != DEFAULTS.DEFAULT_DASHBOARD_V2_APP_PATH
    assert DEFAULTS.DEFAULT_DASHBOARD_V2_APP_PATH.as_posix().endswith(
        "src/trade_bot/dashboard_v2/app.py"
    )
    assert DEFAULTS.DEFAULT_DASHBOARD_V2_PID_PATH != DEFAULTS.DEFAULT_DASHBOARD_PID_PATH
    assert DEFAULTS.DEFAULT_DASHBOARD_V2_LOG_PATH != DEFAULTS.DEFAULT_DASHBOARD_LOG_PATH


def test_dashboard_v2_restores_gated_reference_rail() -> None:
    app_path = Path("src/trade_bot/dashboard_v2/app.py")
    source = app_path.read_text()

    assert "Show quick reference rail" in source
    assert "_install_quick_reference_rail_layout()" in source
    assert "_render_metric_info_rail()" in source
    assert "if show_quick_reference:" in source


def test_dashboard_v2_artifact_service_missing_files_are_empty(tmp_path) -> None:
    assert read_csv_artifact(tmp_path / "missing.csv").empty
    frames = pbo_frames(tmp_path / "missing_pbo_dir")
    assert set(frames) == {"summary", "selection", "stats"}
    assert all(frame.empty for frame in frames.values())


def test_dashboard_v2_card_helper_emits_renderable_html(monkeypatch) -> None:
    captured = {}

    def fake_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        captured["body"] = body
        captured["unsafe"] = unsafe_allow_html

    monkeypatch.setattr(cards.st, "markdown", fake_markdown)
    cards.render_card_grid([("Risk", "Yellow", None, "warning"), ("Score", "0.43")])

    assert captured["unsafe"] is True
    assert '<div class="v2-card v2-card-warning">' in captured["body"]
    assert '<div class="v2-card">' in captured["body"]
    assert 'class="v2-help-dot"' in captured["body"]
    assert "Current traffic-light risk state" in captured["body"]
    assert "\n            <div" not in captured["body"]


def test_dashboard_v2_card_tones_are_domain_aware() -> None:
    assert tones.risk_status_tone("yellow") == "warning"
    assert tones.risk_status_tone("red") == "critical"
    assert tones.risk_status_tone("green") == "success"
    assert tones.portfolio_risk_tone("risk_reduced") == "warning"
    assert tones.portfolio_risk_tone("constraint_breach") == "critical"
    assert tones.portfolio_risk_tone("within_limits") == "success"
    assert tones.instability_tone("ELEVATED") == "warning"
    assert tones.instability_tone(0.62) == "critical"
    assert tones.sleeve_exposure_tone("defensive", 0.77) == "warning"
    assert tones.sleeve_exposure_tone("crypto", 0.0) == "neutral"


def test_dashboard_v2_section_and_chart_helpers_emit_hover_context(monkeypatch) -> None:
    captured: list[tuple[str, str | None]] = []
    rendered_figures: list[go.Figure] = []

    def fake_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        captured.append(("markdown", body))
        assert unsafe_allow_html is True

    def fake_plotly_chart(*args, **kwargs) -> None:
        rendered_figures.append(args[0])
        captured.append(("plotly", str(kwargs.get("use_container_width"))))

    monkeypatch.setattr(cards.st, "markdown", fake_markdown)
    monkeypatch.setattr(cards.st, "plotly_chart", fake_plotly_chart)

    cards.render_section_header("Trade Decision")
    cards.render_chart(go.Figure(layout={"title": {"text": "Selected Market Proxies"}}))

    markdown_bodies = "\n".join(body for kind, body in captured if kind == "markdown")
    assert "Daily operating answer" in markdown_bodies
    assert "Indexes selected price series" in markdown_bodies
    assert "v2-help-popover" in markdown_bodies
    assert any(kind == "plotly" for kind, _body in captured)
    assert rendered_figures[0].layout.title.text == ""


def test_dashboard_v2_native_pages_use_shared_help_wrappers() -> None:
    sources = "\n".join(
        inspect.getsource(module)
        for module in [command_center, monitoring, research, simulation, macro, risk]
    )

    assert "st.subheader(" not in sources
    assert "st.plotly_chart(" not in sources
    assert "render_section_header(" in sources
    assert "render_chart(" in sources


def test_cycle_crisis_playback_figure_shows_horizon_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "crisis": ["sample"] * 4,
            "stage": ["lead_up"] * 4,
            "stage_order": [1] * 4,
            "origin_date": pd.to_datetime(["2026-01-01"] * 2 + ["2026-02-01"] * 2),
            "horizon": ["3m"] * 4,
            "horizon_days": [63] * 4,
            "phase": ["normal_cycle", "liquidation"] * 2,
            "phase_probability": [0.7, 0.3, 0.2, 0.8],
            "dominant_phase": ["normal_cycle", "normal_cycle", "liquidation", "liquidation"],
            "dominant_phase_probability": [0.7, 0.7, 0.8, 0.8],
            "qqq_forward_return": [0.05, 0.05, -0.12, -0.12],
            "spy_forward_return": [0.03, 0.03, -0.08, -0.08],
            "bil_forward_return": [0.01, 0.01, 0.01, 0.01],
            "qqq_forward_drawdown": [-0.02, -0.02, -0.18, -0.18],
            "phase_fit": [True, True, True, True],
        }
    )

    figure = research._crisis_playback_figure(frame)
    trace_names = {trace.name for trace in figure.data}

    assert "QQQ forward return" in trace_names
    assert "QQQ max drawdown" in trace_names
    assert figure.layout.yaxis2.title.text == "3m return / drawdown"


def test_cycle_phase_frontier_figure_highlights_selected_slice() -> None:
    frame = pd.DataFrame(
        {
            "horizon": ["1m", "1m", "3m", "3m"],
            "horizon_days": [21, 21, 63, 63],
            "phase": ["normal_cycle", "acceleration", "normal_cycle", "acceleration"],
            "probability": [0.7, 0.3, 0.4, 0.6],
        }
    )

    figure = research._phase_frontier_figure(
        frame,
        selected_horizon="3m",
        selected_phase="acceleration",
    )
    selected_traces = [trace for trace in figure.data if trace.name == "Selected phase slice"]

    assert len(selected_traces) == 1
    assert selected_traces[0].x[0] == "3m"
    assert float(selected_traces[0].customdata[0]) == 0.6


def test_path_reliability_copy_matches_available_nowcast_horizon() -> None:
    forward_only_intro = research._path_reliability_intro_text(has_nowcast=False)
    forward_only_callout = research._path_reliability_callout_text(has_nowcast=False)
    forward_only_chart_help = research._path_reliability_chart_help_text(has_nowcast=False)

    assert "0M" not in forward_only_intro
    assert "0M" not in forward_only_callout
    assert "0M" not in forward_only_chart_help
    assert "Forward-horizon trust check" in forward_only_callout
    assert "0M measures" in research._path_reliability_callout_text(has_nowcast=True)


def test_cycle_phase_outcome_profile_figure_summarizes_selected_phase() -> None:
    frame = pd.DataFrame(
        {
            "median_forward_return": [0.1, 0.2],
            "median_excess_vs_spy": [0.03, 0.05],
            "median_excess_vs_qqq": [0.01, 0.02],
            "median_forward_drawdown": [-0.04, -0.06],
            "hit_rate_vs_qqq": [0.6, 0.8],
            "origins": [10, 12],
        }
    )

    figure = research._phase_outcome_profile_figure(frame, phase="acceleration")

    assert figure is not None
    assert list(figure.data[0].x) == [
        "Forward return",
        "Excess vs SPY",
        "Excess vs QQQ",
        "Forward drawdown",
    ]
    assert min(figure.data[0].y) < 0
    assert max(figure.data[0].y) > 0


def test_phase_reliability_figure_labels_nowcast_confidence() -> None:
    frame = pd.DataFrame(
        {
            "dominant_phase": ["normal_cycle", "acceleration"],
            "horizon": ["0m", "0m"],
            "horizon_days": [0, 0],
            "origins": [10, 12],
            "phase_fit_rate": [0.45, 0.65],
            "reliability_label": ["mixed_but_useful", "historically_supportive"],
        }
    )

    figure = research._phase_reliability_figure(frame)

    assert figure.layout.xaxis.title.text == "Nowcast confidence"


def test_crisis_playback_figure_uses_confidence_for_nowcast() -> None:
    frame = pd.DataFrame(
        {
            "crisis": ["sample"] * 4,
            "stage": ["lead_up"] * 4,
            "stage_order": [1] * 4,
            "origin_date": pd.to_datetime(["2026-01-01"] * 2 + ["2026-02-01"] * 2),
            "horizon": ["0m"] * 4,
            "horizon_days": [0] * 4,
            "phase": ["normal_cycle", "acceleration"] * 2,
            "phase_probability": [0.7, 0.3, 0.2, 0.8],
            "dominant_phase": ["normal_cycle", "normal_cycle", "acceleration", "acceleration"],
            "dominant_phase_probability": [0.7, 0.7, 0.8, 0.8],
            "qqq_forward_return": [None] * 4,
            "spy_forward_return": [None] * 4,
            "bil_forward_return": [None] * 4,
            "qqq_forward_drawdown": [None] * 4,
            "phase_fit": [True] * 4,
        }
    )

    figure = research._crisis_playback_figure(frame)
    trace_names = {trace.name for trace in figure.data}

    assert "Dominant phase confidence" in trace_names
    assert figure.layout.yaxis2.title.text == "Dominant phase confidence"


def test_launch_and_execution_interaction_guardrails_are_present() -> None:
    launch_source = inspect.getsource(launch_lab._render_launch_lab)
    forward_source = inspect.getsource(forward_test._render_forward_test_and_journal)

    assert "include_summary=False" in launch_source
    assert 'st.form("execution_log_form", enter_to_submit=False)' in forward_source
    assert "min_value=quantity_min" in forward_source
    assert "execution_price_{ticket_key}_{execution_ticker}" in forward_source


def test_forward_test_recalculates_book_alignment_after_journal_changes() -> None:
    source = inspect.getsource(forward_test._render_forward_test_and_journal)

    assert "Recalculate Book Alignment" in source
    assert "Update After-Logged Book Comparison" in source
    assert "_render_after_logged_book_comparison(updated_book_alignment)" in source
    assert "Locked tickets are recommendations only" in source
    assert "warehouse migration refresh" in source
    assert "_rerun_after_journal_mutation" in inspect.getsource(
        forward_test._rerun_after_journal_mutation
    )

    for mutation in [
        "journal.save_decision_snapshot(",
        "journal.log_execution(",
        "journal.update_ticket_status(",
    ]:
        mutation_index = source.index(mutation)
        rerun_index = source.index("_rerun_after_journal_mutation", mutation_index)
        assert mutation_index < rerun_index


def test_after_logged_book_comparison_frame_orders_largest_gap_first() -> None:
    frame = pd.DataFrame(
        [
            {
                "ticker": "BIL",
                "action": "ADD",
                "current_weight": 0.10,
                "scenario_adjusted_weight": 0.20,
                "delta_weight": 0.10,
                "current_notional": 100.0,
                "target_notional": 200.0,
                "delta_notional": 100.0,
                "net_quantity": 1.0,
                "reference_price": 100.0,
            },
            {
                "ticker": "QQQ",
                "action": "REDUCE",
                "current_weight": 0.70,
                "scenario_adjusted_weight": 0.40,
                "delta_weight": -0.30,
                "current_notional": 700.0,
                "target_notional": 400.0,
                "delta_notional": -300.0,
                "net_quantity": 1.4,
                "reference_price": 500.0,
            },
        ]
    )

    comparison = forward_test._after_logged_book_comparison_frame(frame)

    assert list(comparison["ticker"]) == ["QQQ", "BIL"]
    assert list(comparison.columns) == [
        "ticker",
        "action",
        "current_weight",
        "scenario_adjusted_weight",
        "delta_weight",
        "current_notional",
        "target_notional",
        "delta_notional",
        "net_quantity",
        "reference_price",
    ]
    assert comparison.iloc[0]["delta_weight"] == -0.30
