from __future__ import annotations

import inspect
from datetime import date
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
from trade_bot.dashboard_v2.services import runtime as runtime_service
from trade_bot.dashboard_v2.services.artifact_service import (
    defensive_signal_audit_frames,
    pbo_frames,
    prebreak_hindsight_frames,
    read_csv_artifact,
)


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
    assert "include_defensive_judgement=False" in research_source
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
    assert research_source.index("Outcome Frontier") < research_source.index(
        "_render_research_lab("
    )
    assert simulation_source.index('"Strategy simulations"') < simulation_source.index(
        '"Validation"'
    )
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
    prebreak = prebreak_hindsight_frames(tmp_path / "missing_prebreak_dir")
    assert set(prebreak) == {
        "snapshot_signal_panel",
        "signal_predictiveness_rank",
        "action_timing",
        "staged_risk_behavior",
        "late_trigger_mesh",
        "hard_defense_attribution",
        "policy_variant_results",
        "current_best_signal_readout",
        "historical_population_summary",
    }
    assert all(frame.empty for frame in prebreak.values())
    defensive = defensive_signal_audit_frames(tmp_path / "missing_defensive_dir")
    assert set(defensive) == {"current_defensive_exposure", "summary", "scorecards"}
    assert all(frame.empty for frame in defensive.values())


def test_defensive_posture_bridge_joins_focus_strategy_scorecard() -> None:
    exposure = pd.DataFrame(
        [
            {
                "strategy": "focus_strategy",
                "current_defensive_weight": 0.60,
                "current_risk_weight": 0.40,
            }
        ]
    )
    scorecards = pd.DataFrame(
        [
            {
                "strategy": "buy_hold_spy",
                "defensive_benchmark_ticker": "SPY",
                "defensive_judgement_horizon": "1m",
                "defensive_judgement_label": "not_enough_history",
                "defensive_episode_starts": 0,
            },
            {
                "strategy": "focus_strategy",
                "defensive_benchmark_ticker": "QQQ",
                "defensive_judgement_horizon": "1m",
                "defensive_judgement_label": "weak_defensive_signal",
                "defensive_episode_starts": 42,
            },
            {
                "strategy": "focus_strategy",
                "defensive_benchmark_ticker": "SPY",
                "defensive_judgement_horizon": "1m",
                "defensive_judgement_label": "mixed_but_informative",
                "defensive_episode_starts": 42,
            },
        ]
    )

    exposure_row, score_row = research._defensive_posture_rows(exposure, scorecards)

    assert exposure_row["strategy"] == "focus_strategy"
    assert score_row["strategy"] == "focus_strategy"
    assert score_row["defensive_benchmark_ticker"] == "SPY"
    assert score_row["defensive_judgement_label"] == "mixed_but_informative"
    assert research._defensive_label_display(str(score_row["defensive_judgement_label"])) == (
        "Mixed / informative"
    )


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
        captured.append(("plotly", str(kwargs.get("width"))))

    monkeypatch.setattr(cards.st, "markdown", fake_markdown)
    monkeypatch.setattr(cards.st, "plotly_chart", fake_plotly_chart)

    cards.render_section_header("Trade Decision")
    cards.render_chart(go.Figure(layout={"title": {"text": "Selected Market Proxies"}}))

    markdown_bodies = "\n".join(body for kind, body in captured if kind == "markdown")
    assert "Daily operating answer" in markdown_bodies
    assert "Indexes selected price series" in markdown_bodies
    assert "v2-help-popover" in markdown_bodies
    assert any(kind == "plotly" for kind, _body in captured)
    assert ("plotly", "stretch") in captured
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


def test_prebreak_event_behavior_figure_layers_actions_and_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "event_name": ["sample"] * 3,
            "market_date": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15"]),
            "event_break_date": ["2026-01-20"] * 3,
            "risk_budget_multiplier": [0.8, 0.5, 0.3],
            "action_severity_score": [0.0, 0.5, 1.0],
            "forward_spy_return_3m": [0.02, -0.04, -0.08],
            "forward_qqq_return_3m": [0.04, -0.06, -0.12],
            "forward_smh_return_3m": [0.05, -0.08, -0.18],
            "forward_min_max_drawdown_3m": [-0.03, -0.12, -0.22],
        }
    )

    figure = research._prebreak_event_behavior_figure(frame)
    trace_names = {trace.name for trace in figure.data}

    assert "Risk budget multiplier" in trace_names
    assert "Action severity" in trace_names
    assert "Worst 3m drawdown" in trace_names
    assert any(shape.type == "line" for shape in figure.layout.shapes)


def test_prebreak_signal_rank_figure_marks_current_risk_reads() -> None:
    frame = pd.DataFrame(
        {
            "signal": ["health_qqq_drawdown", "cycle_component_qqq_unwind"],
            "predictive_score": [0.35, 0.33],
            "risk_direction": ["higher_is_riskier", "lower_is_riskier"],
            "current_risk_read": ["elevated", "contained"],
            "historical_percentile": [0.95, 0.2],
            "spearman_to_break_severity": [0.28, -0.27],
            "event_auc": [0.71, 0.30],
            "high_minus_low_break_severity": [0.05, -0.04],
        }
    )

    figure = research._prebreak_signal_rank_figure(frame)

    assert len(figure.data) == 1
    assert "Market: Qqq Drawdown" in set(figure.data[0].y)
    assert "#ef4444" in list(figure.data[0].marker.color)


def test_prebreak_population_summary_separates_origins_from_conservative_clusters() -> None:
    frame = pd.DataFrame(
        {
            "market_date": ["2020-01-08", "2020-01-15", "2020-04-15", "2020-07-15"],
            "event_name": ["sample_break", "sample_break", "", ""],
            "population_role": [
                "event_window",
                "event_window",
                "historical_control",
                "historical_control",
            ],
            "population_cluster": [
                "event:sample_break",
                "event:sample_break",
                "control:2020Q2",
                "control:2020Q3",
            ],
            "break_severity_3m": [0.2, 0.1, 0.0, float("nan")],
        }
    )

    summary = research._prebreak_population_summary(frame)

    assert summary == {
        "origins": 4,
        "mature_outcomes": 3,
        "named_events": 1,
        "event_origins": 2,
        "ordinary_controls": 2,
        "population_clusters": 3,
        "date_range": "2020-01-08 to 2020-07-15",
    }


def test_prebreak_event_selector_options_filter_rollups_and_default_alias() -> None:
    signal_panel = pd.DataFrame(
        {
            "event_name": [
                "gfc_credit_bubble_peak",
                "ALL_SEVERE_0-21d",
                "covid_crash_peak",
            ]
        }
    )
    action_timing = pd.DataFrame(
        {
            "event_name": [
                "q4_2018_liquidity_break",
                "gfc_credit_bubble_peak",
            ]
        }
    )

    options = research._prebreak_event_options(signal_panel, action_timing)

    assert options == [
        "q4_2018_liquidity_break",
        "gfc_credit_bubble_peak",
        "covid_crash_peak",
    ]
    assert (
        research._default_prebreak_event_for_crisis("global_financial_crisis", options)
        == "gfc_credit_bubble_peak"
    )
    assert (
        research._default_prebreak_event_for_crisis("unknown_cycle_window", options)
        == "q4_2018_liquidity_break"
    )


def test_prebreak_event_selector_renders_below_selected_behavior_heading(monkeypatch) -> None:
    rendered: list[str] = []
    signal_panel = pd.DataFrame({"event_name": ["covid_crash_peak"]})

    monkeypatch.setattr(
        research,
        "render_section_header",
        lambda title, **_kwargs: rendered.append(f"heading:{title}"),
    )
    monkeypatch.setattr(research, "render_callout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(research, "render_card_grid", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        research,
        "_select_prebreak_event",
        lambda **_kwargs: rendered.append("selector") or "covid_crash_peak",
    )
    monkeypatch.setattr(
        research,
        "_render_selected_prebreak_event_behavior",
        lambda **_kwargs: rendered.append("behavior"),
    )
    monkeypatch.setattr(
        research,
        "_render_prebreak_margin_experiment",
        lambda **_kwargs: rendered.append("margin"),
    )

    research._render_prebreak_hindsight_layers(
        selected_crisis="covid_crash",
        prebreak={"snapshot_signal_panel": signal_panel},
        defensive_audit={},
    )

    assert rendered == [
        "heading:Pre-Break Behavior And Early Warning",
        "heading:Historical Sample Population",
        "heading:Selected Crisis Trade-Bot Behavior",
        "selector",
        "behavior",
        "margin",
    ]


def test_late_trigger_interpretation_explains_hindsight_and_percentage_points() -> None:
    read = research._late_trigger_interpretation(
        pd.Series(
            {
                "trigger_days_before_break": 45,
                "hard_defense_lead_cut_days": 41,
                "missed_severe_label_share_if_gated": 0.0,
                "mean_candidate_risk_budget_lift": 0.267262,
                "median_forward_drawdown_when_lifted": -0.094432,
            }
        )
    )

    assert "Selected-event hindsight" in read
    assert "26.73 pp" in read
    assert "9.44%" in read
    assert research._fmt_pp(0.267262) == "26.73 pp"


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


def test_crisis_playback_market_figure_shows_ticker_paths_and_drawdowns() -> None:
    frame = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "stage": ["lead_up", "unwind"],
            "stage_order": [1, 2],
            "qqq_playback_index": [1.0, 0.85],
            "qqq_playback_drawdown": [0.0, -0.15],
            "qqq_playback_close": [100.0, 85.0],
            "spy_playback_index": [1.0, 0.92],
            "spy_playback_drawdown": [0.0, -0.08],
            "spy_playback_close": [100.0, 92.0],
        }
    )

    figure = research._crisis_playback_market_figure(frame, tickers=["QQQ", "SPY"])
    trace_names = {trace.name for trace in figure.data}

    assert {"QQQ indexed price", "QQQ drawdown", "SPY indexed price", "SPY drawdown"}.issubset(
        trace_names
    )
    assert figure.layout.yaxis.title.text == "Index"
    assert figure.layout.yaxis2.title.text == "Drawdown"


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
