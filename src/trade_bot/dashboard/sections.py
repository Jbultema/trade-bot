from __future__ import annotations

from typing import Any

import pandas as pd

from trade_bot.dashboard.command_center import _render_command_center
from trade_bot.dashboard.forward_test import _render_forward_test_and_journal
from trade_bot.dashboard.launch_lab import _render_launch_lab
from trade_bot.dashboard.monitoring import _render_monitoring
from trade_bot.dashboard.news_macro import _render_news_and_macro
from trade_bot.dashboard.performance import _render_performance
from trade_bot.dashboard.research_lab import _render_research_lab
from trade_bot.dashboard.risk_scenarios import _render_risk_and_scenarios
from trade_bot.dashboard.simulation_lab import _render_simulation_lab
from trade_bot.research.baselines import BaselineRun
from trade_bot.trading.book_alignment import BookAlignmentRun
from trade_bot.trading.journal import TradeJournal


def _render_dashboard_section(
    section: str,
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    journal: TradeJournal,
    experiment_scorecards: pd.DataFrame,
    experiment_regimes: pd.DataFrame,
    experiment_walk_forward: pd.DataFrame,
    experiment_candidates: pd.DataFrame,
    decision_sanity_impacts: pd.DataFrame,
    warehouse_path: str,
    artifact_dir: str = "",
    job_log_dir: str = "",
    book_alignment: BookAlignmentRun | None = None,
) -> None:
    if section == "Command Center":
        _render_command_center(baseline_run, book_alignment=book_alignment)
    elif section == "Risk & Scenarios":
        _render_risk_and_scenarios(
            baseline_run,
            run_store_path=warehouse_path,
            artifact_dir=artifact_dir,
            job_log_dir=job_log_dir,
        )
    elif section == "Simulation Lab":
        _render_simulation_lab(
            bot_config,
            baseline_run,
            experiment_scorecards,
            warehouse_path=warehouse_path,
        )
    elif section == "Launch Lab":
        _render_launch_lab(bot_config, baseline_run, experiment_scorecards)
    elif section == "Monitoring":
        _render_monitoring(warehouse_path)
    elif section == "Research Lab":
        _render_research_lab(
            bot_config,
            baseline_run,
            experiment_scorecards,
            experiment_regimes,
            experiment_walk_forward,
            experiment_candidates,
            decision_sanity_impacts,
        )
    elif section == "News & Macro":
        _render_news_and_macro(baseline_run)
    elif section == "Performance":
        _render_performance(
            baseline_run,
            bot_config=bot_config,
            experiment_scorecards=experiment_scorecards,
        )
    elif section == "Forward Test":
        _render_forward_test_and_journal(
            journal,
            baseline_run,
            bot_config=bot_config,
            warehouse_path=warehouse_path,
        )
