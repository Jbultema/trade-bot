from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from trade_bot.dashboard_v2.services.runtime import DashboardRuntime


@dataclass(frozen=True)
class DashboardRoute:
    key: str
    label: str
    lane: str
    title: str
    question: str
    runtime: str
    runtime_note: str
    render: Callable[[DashboardRuntime], None]


def routes() -> tuple[DashboardRoute, ...]:
    from trade_bot.dashboard_v2.pages.command_center import render_today_page
    from trade_bot.dashboard_v2.pages.full_workbench import (
        render_forward_test_page,
        render_launch_page,
        render_performance_page,
        render_risk_scenarios_page,
    )
    from trade_bot.dashboard_v2.pages.macro import render_macro_page
    from trade_bot.dashboard_v2.pages.monitoring import render_monitoring_page
    from trade_bot.dashboard_v2.pages.research import render_research_page
    from trade_bot.dashboard_v2.pages.simulation import render_simulation_page

    return (
        DashboardRoute(
            key="today",
            label="Today",
            lane="Operate",
            title="Today",
            question="What is the system asking me to do now?",
            runtime="Fast",
            runtime_note="Reads the latest snapshot and only loads trends/raw tables when requested.",
            render=render_today_page,
        ),
        DashboardRoute(
            key="monitoring",
            label="Monitoring",
            lane="Decide",
            title="Monitoring",
            question="Are monitored strategies proving out after their start dates?",
            runtime="Fast by default",
            runtime_note="Loads active windows first; forward trends and full controls are explicit subviews.",
            render=render_monitoring_page,
        ),
        DashboardRoute(
            key="launch",
            label="Launch",
            lane="Decide",
            title="Launch Lab",
            question="Should new money start, wait, or ramp into a selected strategy?",
            runtime="On demand",
            runtime_note="Compatibility page; use internal view pickers before loading aggregate reads.",
            render=render_launch_page,
        ),
        DashboardRoute(
            key="research",
            label="Research",
            lane="Research",
            title="Research",
            question="Which candidates still deserve belief?",
            runtime="Fast by default",
            runtime_note="Loads scorecard summaries first; full workbench and raw artifacts are gated.",
            render=render_research_page,
        ),
        DashboardRoute(
            key="simulation",
            label="Simulation",
            lane="Research",
            title="Simulation",
            question="What future range and validation quality should I expect?",
            runtime="Fast by default",
            runtime_note="Loads persisted validation summaries first; path engines remain explicit.",
            render=render_simulation_page,
        ),
        DashboardRoute(
            key="risk",
            label="Risk",
            lane="Operate",
            title="Risk & Scenarios",
            question="Why is exposure being capped, expanded, or left alone?",
            runtime="Medium",
            runtime_note="Compatibility page over the existing risk-scenario renderer.",
            render=render_risk_scenarios_page,
        ),
        DashboardRoute(
            key="macro",
            label="Macro",
            lane="Research",
            title="News & Macro",
            question="What context is active, and what is only watch-only narrative?",
            runtime="Fast by default",
            runtime_note="Uses snapshot price and macro frames first; the full macro workbench is explicit.",
            render=render_macro_page,
        ),
        DashboardRoute(
            key="performance",
            label="Performance",
            lane="Research",
            title="Performance",
            question="Did the system work in the selected historical window?",
            runtime="Medium",
            runtime_note="Compatibility page over the existing performance renderer.",
            render=render_performance_page,
        ),
        DashboardRoute(
            key="forward",
            label="Forward Test",
            lane="Operate",
            title="Forward Test",
            question="What did I lock, execute, and record?",
            runtime="Medium",
            runtime_note="Compatibility page over the existing journal and ticket renderer.",
            render=render_forward_test_page,
        ),
    )


def route_by_key(route_key: str) -> DashboardRoute:
    route_map = {route.key: route for route in routes()}
    return route_map.get(route_key, route_map["today"])
