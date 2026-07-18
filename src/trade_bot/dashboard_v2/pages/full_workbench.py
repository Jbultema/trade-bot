from __future__ import annotations

from trade_bot.dashboard.forward_test import _render_forward_test_and_journal
from trade_bot.dashboard.launch_lab import _render_launch_lab
from trade_bot.dashboard.news_macro import _render_news_and_macro
from trade_bot.dashboard.performance import _render_performance
from trade_bot.dashboard.risk_scenarios import _render_risk_and_scenarios
from trade_bot.dashboard_v2.components.cards import render_callout
from trade_bot.dashboard_v2.services.experiment_service import scorecards
from trade_bot.dashboard_v2.services.runtime import DashboardRuntime, render_book_selector


def render_risk_scenarios_page(runtime: DashboardRuntime) -> None:
    render_callout(
        "Risk & Scenarios is using the full renderer in V2. It reads snapshot data and saved trend tables.",
        heavy=True,
    )
    _render_risk_and_scenarios(
        runtime.baseline_run,
        run_store_path=runtime.paths.run_store_path,
        artifact_dir=runtime.paths.artifact_dir,
        job_log_dir=runtime.paths.job_log_dir,
    )


def render_news_macro_page(runtime: DashboardRuntime) -> None:
    render_callout(
        "News & Macro is using the full renderer in V2. Narrative and macro diagnostics are artifact-backed.",
        heavy=True,
    )
    _render_news_and_macro(
        runtime.baseline_run,
        run_store_path=runtime.paths.run_store_path,
        artifact_dir=runtime.paths.artifact_dir,
        job_log_dir=runtime.paths.job_log_dir,
    )


def render_launch_page(runtime: DashboardRuntime) -> None:
    render_callout(
        "Launch Lab uses the full workbench renderer. Internal view pickers gate the heavier aggregate reads.",
        heavy=True,
    )
    _render_launch_lab(
        runtime.bot_config,
        runtime.baseline_run,
        scorecards(),
        warehouse_path=str(runtime.paths.run_store_path),
    )


def render_performance_page(runtime: DashboardRuntime) -> None:
    _render_performance(
        runtime.baseline_run,
        bot_config=runtime.bot_config,
        experiment_scorecards=scorecards(),
    )


def render_forward_test_page(runtime: DashboardRuntime) -> None:
    _render_forward_test_and_journal(
        runtime.journal,
        runtime.baseline_run,
        bot_config=runtime.bot_config,
        warehouse_path=str(runtime.paths.run_store_path),
        selected_book=runtime.selected_book,
        book_selector=lambda: render_book_selector(
            runtime.paths.journal_path,
            baseline_run=runtime.baseline_run,
            bot_config=runtime.bot_config,
        ).selected_book,
    )
