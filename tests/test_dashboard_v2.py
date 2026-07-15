from __future__ import annotations

import inspect

from trade_bot import DEFAULTS
from trade_bot.dashboard_v2 import routes
from trade_bot.dashboard_v2.pages import command_center, monitoring, research, simulation
from trade_bot.dashboard_v2.services.artifact_service import pbo_frames, read_csv_artifact


def test_dashboard_v2_routes_cover_core_workflows() -> None:
    route_list = routes.routes()
    labels = [route.label for route in route_list]

    assert labels == [
        "Today",
        "Monitoring",
        "Launch",
        "Research",
        "Simulation",
        "Risk",
        "Macro",
        "Performance",
        "Forward Test",
    ]
    assert {route.lane for route in route_list} == {"Operate", "Decide", "Research"}
    assert routes.route_by_key("missing").key == "today"
    assert routes.route_by_key("research").runtime == "Fast by default"
    assert routes.route_by_key("simulation").runtime == "Fast by default"


def test_dashboard_v2_native_pages_are_summary_first() -> None:
    today_source = inspect.getsource(command_center.render_today_page)
    monitoring_source = inspect.getsource(monitoring.render_monitoring_page)
    research_source = inspect.getsource(research.render_research_page)
    simulation_source = inspect.getsource(simulation.render_simulation_page)

    assert "st.tabs(" not in today_source
    assert "st.tabs(" not in monitoring_source
    assert "st.tabs(" not in research_source
    assert "st.tabs(" not in simulation_source
    assert today_source.index("Decision") < today_source.index("load_snapshot_trend_frames")
    assert monitoring_source.index("Readout") < monitoring_source.index("_render_monitoring(")
    assert research_source.index("Leaderboard") < research_source.index("_render_research_lab(")
    assert simulation_source.index("Validation summary") < simulation_source.index(
        "_render_simulation_lab("
    )


def test_dashboard_v2_uses_separate_entrypoint_and_process_defaults() -> None:
    assert DEFAULTS.DEFAULT_DASHBOARD_APP_PATH != DEFAULTS.DEFAULT_DASHBOARD_V2_APP_PATH
    assert DEFAULTS.DEFAULT_DASHBOARD_V2_APP_PATH.as_posix().endswith(
        "src/trade_bot/dashboard_v2/app.py"
    )
    assert DEFAULTS.DEFAULT_DASHBOARD_V2_PID_PATH != DEFAULTS.DEFAULT_DASHBOARD_PID_PATH
    assert DEFAULTS.DEFAULT_DASHBOARD_V2_LOG_PATH != DEFAULTS.DEFAULT_DASHBOARD_LOG_PATH


def test_dashboard_v2_artifact_service_missing_files_are_empty(tmp_path) -> None:
    assert read_csv_artifact(tmp_path / "missing.csv").empty
    frames = pbo_frames(tmp_path / "missing_pbo_dir")
    assert set(frames) == {"summary", "selection", "stats"}
    assert all(frame.empty for frame in frames.values())

