from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from trade_bot.config import configured_tickers, load_config
from trade_bot.data.fred_data import load_fred_catalog, load_or_fetch_fred_data
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.DEFAULTS import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DASHBOARD_APP_PATH,
    DEFAULT_DASHBOARD_LOG_PATH,
    DEFAULT_DASHBOARD_PID_PATH,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_DASHBOARD_V2_APP_PATH,
    DEFAULT_DASHBOARD_V2_LOG_PATH,
    DEFAULT_DASHBOARD_V2_PID_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_EXTERNAL_MACRO_ALIGNMENT_DIR,
    DEFAULT_EXTERNAL_MACRO_TRANSCRIPT_DIR,
    DEFAULT_FACTOR_ATTRIBUTION_FACTOR_SPECS,
    DEFAULT_FORWARD_SIMULATION_BLOCK_DAYS,
    DEFAULT_FORWARD_SIMULATION_PATHS,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_HORIZONS,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_HIGH,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_LOW,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_MIN_TRAIN_DAYS,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_ORIGIN_FREQUENCY,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MACRO_PATH,
    DEFAULT_ML_DIAGNOSTICS_DIR,
    DEFAULT_MONITORING_COHORT_START_DATE,
    DEFAULT_MONITORING_TOP_N,
    DEFAULT_NEWS_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_RESET_EXPERIMENTS_DIR,
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
    DEFAULT_SIGNAL_EVIDENCE_DIR,
    DEFAULT_SIMULATION_REFERENCE_STRATEGIES,
    DEFAULT_SNAPSHOT_BACKFILL_DAILY_TAIL_DAYS,
    DEFAULT_SNAPSHOT_BACKFILL_MARKET_CLOSE_UTC_HOUR,
    DEFAULT_SNAPSHOT_BACKFILL_YEARS,
    DEFAULT_SNAPSHOT_RETENTION_KEEP_LATEST,
    DEFAULT_SNAPSHOT_RETENTION_KEEP_PER_MARKET_DATE,
    DEFAULT_SNAPSHOT_RETENTION_KEEP_RECENT_MARKET_DAYS,
    DEFAULT_SNAPSHOT_RETENTION_KEEP_WEEKLY_OLDER,
    DEFAULT_SNAPSHOT_RETENTION_WEEKLY_FREQUENCY,
    DEFAULT_STRATEGY_SOURCE_AUDIT_DIR,
)
from trade_bot.ml.diagnostics import run_ml_diagnostics
from trade_bot.reporting.report import write_baseline_report
from trade_bot.research.backtest_pbo import (
    pbo_candidate_tickers,
    run_backtest_pbo_gauntlet,
)
from trade_bot.research.backtest_qc import DEFAULT_QC_STRATEGY, run_backtest_qc_gauntlet
from trade_bot.research.baselines import (
    run_configured_baselines,
    run_configured_baselines_from_frames,
)
from trade_bot.research.cycle_tracker import (
    DEFAULT_PHASE_MIN_TRAIN_DAYS,
    DEFAULT_PHASE_VALIDATION_STEP_DAYS,
    run_cycle_tracker,
)
from trade_bot.research.defensive_judgement import write_defensive_judgement_report
from trade_bot.research.entry_date_analysis import build_entry_date_analysis
from trade_bot.research.experiment_monitor import (
    load_experiment_candidates,
    load_experiment_scorecards,
)
from trade_bot.research.experiments import run_experiment_iteration
from trade_bot.research.external_macro import (
    DEFAULT_42MACRO_HANDLE,
    compare_42macro_to_trade_bot,
    import_42macro_transcript_files,
    score_macro_tradebot_outcomes,
    sync_42macro_transcripts,
    write_missing_42macro_transcript_priority,
)
from trade_bot.research.forward_simulation import (
    ForwardSimulationValidationConfig,
    rolling_origin_simulation_backtest,
    rolling_origin_strategy_rank_validation,
    summarize_simulation_validation,
    summarize_simulation_validation_by_horizon,
    summarize_strategy_rank_validation,
)
from trade_bot.research.leadership_diagnostics import (
    leadership_candidate_tickers,
    run_leadership_diagnostics,
)
from trade_bot.research.operating_history import (
    DEFAULT_OPERATING_HISTORY_PRIMARY_STRATEGY,
    DEFAULT_OPERATING_HISTORY_SOURCE,
    reconstruct_operating_history,
)
from trade_bot.research.prebreak_hindsight import (
    DEFAULT_POSTBREAK_FOLLOWTHROUGH_DAYS,
    DEFAULT_PREBREAK_HORIZON_DAYS,
    DEFAULT_PREBREAK_LOOKBACK_DAYS,
    DEFAULT_PREBREAK_OUTPUT_DIR,
    DEFAULT_PREBREAK_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_PREBREAK_RUN_STORE_DB_PATH,
    DEFAULT_PREBREAK_RUN_STORE_JOB_LOG_DIR,
    DEFAULT_PREBREAK_WEEKLY_FREQUENCY,
    analyze_prebreak_hindsight,
    build_prebreak_snapshot_plan,
    write_prebreak_hindsight_outputs,
)
from trade_bot.research.scenario_history import (
    clean_scenario_history,
    reconstruct_scenario_history_from_prices,
    scenario_history_from_snapshots,
    write_scenario_history,
)
from trade_bot.research.signal_evidence import (
    build_signal_family_evidence,
    build_signal_family_marginal_tests,
    tag_scorecard_signal_families,
)
from trade_bot.research.snapshot_backfill import (
    SnapshotBackfillPlan,
    build_snapshot_backfill_plan,
    snapshot_created_at_for_market_date,
)
from trade_bot.research.strategy_source_audit import write_strategy_source_audit
from trade_bot.research.upside_capture import (
    DEFAULT_UPSIDE_CAPTURE_OUTPUT_DIR,
    run_upside_capture_lab,
)
from trade_bot.storage.run_store import RunStore, SnapshotManifest
from trade_bot.storage.warehouse import TradingWarehouse, WarehouseMigrationResult

app = typer.Typer(no_args_is_help=True)
console = Console()
DEFAULT_SIMULATION_VALIDATION_REFERENCE_STRATEGIES = ",".join(
    name for name, _label in DEFAULT_SIMULATION_REFERENCE_STRATEGIES
)
DEFAULT_STREAMLIT_FILE_WATCHER_TYPE = "none"


@app.command()
def fetch_prices(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
) -> None:
    bot_config = load_config(config)
    prices = load_or_fetch_yahoo_prices(
        configured_tickers(bot_config),
        start=bot_config.data.start,
        end=bot_config.data.end,
        cache_dir=bot_config.data.cache_dir,
        adjusted=bot_config.data.adjusted,
        refresh=refresh,
    )
    console.print(
        f"Loaded {prices.shape[1]} tickers and {prices.shape[0]} rows "
        f"from {prices.index.min().date()} to {prices.index.max().date()}."
    )


@app.command()
def run_baselines(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    events: Annotated[Path, typer.Option("--events")] = DEFAULT_EVENTS_PATH,
    macro: Annotated[Path, typer.Option("--macro")] = DEFAULT_MACRO_PATH,
    news: Annotated[Path, typer.Option("--news")] = DEFAULT_NEWS_PATH,
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
    refresh_macro: Annotated[bool, typer.Option("--refresh-macro")] = False,
    refresh_news: Annotated[bool, typer.Option("--refresh-news")] = False,
    report_path: Annotated[Path, typer.Option("--report-path")] = DEFAULT_REPORT_PATH,
) -> None:
    bot_config = load_config(config)
    baseline_run = run_configured_baselines(
        bot_config,
        refresh_data=refresh_data,
        refresh_macro=refresh_macro,
        refresh_news=refresh_news,
        event_config_path=events,
        macro_config_path=macro,
        news_config_path=news,
    )
    write_baseline_report(
        baseline_run.results,
        baseline_run.metrics,
        baseline_run.window_summary,
        baseline_run.calendar_returns,
        baseline_run.current_state,
        baseline_run.event_risk,
        baseline_run.news_monitor,
        baseline_run.signal_inclusion,
        baseline_run.trade_decision,
        report_path,
    )

    console.print(
        f"[bold]Current risk status:[/bold] {baseline_run.current_state.risk_status.upper()} "
        f"({baseline_run.current_state.risk_score:.2f})"
    )
    console.print(baseline_run.current_state.risk_summary)
    console.print(
        f"[bold]Signal coverage:[/bold] {baseline_run.prices.shape[1]:,} market proxies, "
        f"{len(baseline_run.macro_catalog):,} configured macro series, "
        f"{baseline_run.macro_data.shape[1]:,} loaded macro series."
    )

    alert_table = Table(title="Current Strategy Alerts")
    for column in ["strategy", "priority", "action", "latest_position", "trade_alert"]:
        alert_table.add_column(column)
    for _, row in baseline_run.current_state.strategy_alerts.iterrows():
        alert_table.add_row(
            str(row["strategy"]),
            str(row["priority"]),
            str(row["action"]),
            str(row["latest_position"]),
            str(row["trade_alert"]),
        )
    console.print(alert_table)

    if not baseline_run.trade_decision.summary.empty:
        trade_row = baseline_run.trade_decision.summary.iloc[0]
        console.print(
            "[bold]Scenario-adjusted trade decision:[/bold] "
            f"{trade_row['recommended_action']} | "
            f"{trade_row['scenario_adjusted_position']}"
        )
        console.print(str(trade_row["human_explanation"]))

        position_table = Table(title="Scenario-Adjusted Position Bridge")
        for column in [
            "ticker",
            "current_weight",
            "scenario_adjusted_weight",
            "delta_weight",
            "action",
        ]:
            position_table.add_column(column)
        for _, row in baseline_run.trade_decision.position_plan.iterrows():
            position_table.add_row(
                str(row["ticker"]),
                _format_optional_percent(row["current_weight"]),
                _format_optional_percent(row["scenario_adjusted_weight"]),
                _format_optional_percent(row["delta_weight"]),
                str(row["action"]),
            )
        console.print(position_table)

        if (
            baseline_run.portfolio_risk is not None
            and not baseline_run.portfolio_risk.summary.empty
        ):
            risk_row = baseline_run.portfolio_risk.summary.iloc[0]
            console.print(
                "[bold]Portfolio risk engine:[/bold] "
                f"{risk_row['portfolio_risk_level']} | "
                f"risk multiplier {risk_row['portfolio_risk_multiplier']:.2f} | "
                f"ES95 {risk_row['post_expected_shortfall_95']:.2%} | "
                f"max stress loss {risk_row['post_max_stress_loss']:.2%}"
            )

    table = Table(title="Baseline Backtests")
    for column in [
        "strategy",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "average_turnover",
    ]:
        table.add_column(column)

    for name, row in baseline_run.metrics.iterrows():
        table.add_row(
            name,
            f"{row['cagr']:.2%}",
            f"{row['sharpe']:.2f}",
            f"{row['sortino']:.2f}",
            f"{row['max_drawdown']:.2%}",
            f"{row['calmar']:.2f}",
            f"{row['average_turnover']:.2%}",
        )
    console.print(table)

    if not baseline_run.window_summary.empty:
        window_table = Table(title="Rolling Window Diagnostics")
        for column in [
            "strategy",
            "window",
            "median_cagr",
            "worst_cagr",
            "worst_drawdown",
            "positive_window_rate",
            "median_calmar",
        ]:
            window_table.add_column(column)

        for (name, window), row in baseline_run.window_summary.iterrows():
            window_table.add_row(
                name,
                window,
                f"{row['median_cagr']:.2%}",
                f"{row['worst_cagr']:.2%}",
                f"{row['worst_drawdown']:.2%}",
                f"{row['positive_window_rate']:.2%}",
                f"{row['median_calmar']:.2f}",
            )
        console.print(window_table)

    if not baseline_run.event_risk.event_summary.empty:
        event_table = Table(title="Event-Risk Summary")
        for column in [
            "event_name",
            "window",
            "market_mode",
            "risk_asset_return",
            "oil_complex_return",
            "primary_strategy_return",
        ]:
            event_table.add_column(column)

        recent_events = baseline_run.event_risk.event_summary[
            baseline_run.event_risk.event_summary["window"].isin(["post_5d", "post_21d"])
        ].tail(12)
        for _, row in recent_events.iterrows():
            event_table.add_row(
                str(row["event_name"]),
                str(row["window"]),
                str(row["market_mode"]),
                _format_optional_percent(row["risk_asset_return"]),
                _format_optional_percent(row["oil_complex_return"]),
                _format_optional_percent(row["primary_strategy_return"]),
            )
        console.print(event_table)

    if not baseline_run.news_monitor.triage.empty:
        news_table = Table(title="News Intake Triage")
        for column in [
            "title",
            "source",
            "category",
            "phase",
            "urgency_score",
            "activation_status",
        ]:
            news_table.add_column(column)

        for _, row in baseline_run.news_monitor.triage.head(10).iterrows():
            news_table.add_row(
                str(row["title"])[:80],
                str(row["source"]),
                str(row["category"]),
                str(row["phase"]),
                f"{float(row['urgency_score']):.2f}",
                str(row["activation_status"]),
            )
        console.print(news_table)

    if not baseline_run.signal_inclusion.summary.empty:
        inclusion_table = Table(title="Signal Inclusion Tests")
        for column in [
            "signal_group",
            "decision",
            "latest_pressure_state",
            "delta_cagr",
            "max_drawdown_improvement",
            "delta_calmar",
        ]:
            inclusion_table.add_column(column)

        for _, row in baseline_run.signal_inclusion.summary.head(12).iterrows():
            inclusion_table.add_row(
                str(row["signal_group"]),
                str(row["decision"]),
                str(row["latest_pressure_state"]),
                _format_optional_percent(row["delta_cagr"]),
                _format_optional_percent(row["max_drawdown_improvement"]),
                (
                    f"{float(row['delta_calmar']):.2f}"
                    if row["delta_calmar"] == row["delta_calmar"]
                    else "n/a"
                ),
            )
        console.print(inclusion_table)

    console.print(f"Report written to {report_path}")


@app.command("build-snapshot")
def build_snapshot_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    events: Annotated[Path, typer.Option("--events")] = DEFAULT_EVENTS_PATH,
    macro: Annotated[Path, typer.Option("--macro")] = DEFAULT_MACRO_PATH,
    news: Annotated[Path, typer.Option("--news")] = DEFAULT_NEWS_PATH,
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
    refresh_macro: Annotated[bool, typer.Option("--refresh-macro")] = False,
    refresh_news: Annotated[bool, typer.Option("--refresh-news")] = False,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    report_path: Annotated[Path, typer.Option("--report-path")] = DEFAULT_REPORT_PATH,
    write_report: Annotated[bool, typer.Option("--write-report/--no-write-report")] = True,
    job_id: Annotated[str | None, typer.Option("--job-id")] = None,
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    if job_id:
        run_store.mark_job_running(job_id)

    try:
        bot_config = load_config(config)
        baseline_run = run_configured_baselines(
            bot_config,
            refresh_data=refresh_data,
            refresh_macro=refresh_macro,
            refresh_news=refresh_news,
            event_config_path=events,
            macro_config_path=macro,
            news_config_path=news,
        )
        manifest = run_store.save_snapshot(
            baseline_run,
            config_path=config,
            events_path=events,
            macro_path=macro,
            news_path=news,
            refresh_data=refresh_data,
            refresh_macro=refresh_macro,
            refresh_news=refresh_news,
        )
        if write_report:
            write_baseline_report(
                baseline_run.results,
                baseline_run.metrics,
                baseline_run.window_summary,
                baseline_run.calendar_returns,
                baseline_run.current_state,
                baseline_run.event_risk,
                baseline_run.news_monitor,
                baseline_run.signal_inclusion,
                baseline_run.trade_decision,
                report_path,
            )
        if job_id:
            run_store.mark_job_completed(job_id, manifest.run_id)
    except Exception as error:
        if job_id:
            run_store.mark_job_failed(job_id, str(error))
        raise

    _print_snapshot_manifest(manifest)


@app.command("run-daily-update")
def run_daily_update_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    events: Annotated[Path, typer.Option("--events")] = DEFAULT_EVENTS_PATH,
    macro: Annotated[Path, typer.Option("--macro")] = DEFAULT_MACRO_PATH,
    news: Annotated[Path, typer.Option("--news")] = DEFAULT_NEWS_PATH,
    refresh_data: Annotated[bool, typer.Option("--refresh-data/--cached-data")] = True,
    refresh_macro: Annotated[bool, typer.Option("--refresh-macro/--cached-macro")] = True,
    refresh_news: Annotated[bool, typer.Option("--refresh-news/--cached-news")] = True,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    report_path: Annotated[Path, typer.Option("--report-path")] = DEFAULT_REPORT_PATH,
    experiment_dir: Annotated[Path, typer.Option("--experiment-dir")] = DEFAULT_EXPERIMENTS_DIR,
    journal: Annotated[Path, typer.Option("--journal")] = DEFAULT_JOURNAL_PATH,
    migrate_warehouse: Annotated[
        bool,
        typer.Option("--migrate-warehouse/--skip-warehouse"),
    ] = True,
    paper_valuation: Annotated[
        bool,
        typer.Option("--paper-valuation/--skip-paper-valuation"),
    ] = True,
    job_id: Annotated[str | None, typer.Option("--job-id")] = None,
) -> None:
    """Run the full daily operating refresh for the dashboard."""
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    if job_id:
        run_store.mark_job_running(job_id)

    warehouse = TradingWarehouse(store)
    migration_results: list[WarehouseMigrationResult] = []
    registry_rows = 0
    valuation_rows = 0
    effective_experiment_dir = _active_experiment_dir(experiment_dir)
    try:
        bot_config = load_config(config)
        baseline_run = run_configured_baselines(
            bot_config,
            refresh_data=refresh_data,
            refresh_macro=refresh_macro,
            refresh_news=refresh_news,
            event_config_path=events,
            macro_config_path=macro,
            news_config_path=news,
        )
        manifest = run_store.save_snapshot(
            baseline_run,
            config_path=config,
            events_path=events,
            macro_path=macro,
            news_path=news,
            refresh_data=refresh_data,
            refresh_macro=refresh_macro,
            refresh_news=refresh_news,
        )
        write_baseline_report(
            baseline_run.results,
            baseline_run.metrics,
            baseline_run.window_summary,
            baseline_run.calendar_returns,
            baseline_run.current_state,
            baseline_run.event_risk,
            baseline_run.news_monitor,
            baseline_run.signal_inclusion,
            baseline_run.trade_decision,
            report_path,
        )
        registry_rows = warehouse.refresh_strategy_registry_from_snapshot(
            baseline_run,
            run_id=manifest.run_id,
            market_date=manifest.market_date,
        )
        if migrate_warehouse:
            migration_results.extend(warehouse.migrate_experiment_outputs(effective_experiment_dir))
            migration_results.extend(warehouse.migrate_journal_sqlite(journal))
        if paper_valuation:
            valuation_rows = warehouse.save_daily_valuations_from_snapshot(
                baseline_run,
                market_date=manifest.market_date,
                execution=bot_config.execution,
            )
        if job_id:
            run_store.mark_job_completed(job_id, manifest.run_id)
    except Exception as error:
        if job_id:
            run_store.mark_job_failed(job_id, str(error))
        raise

    _print_snapshot_manifest(manifest)
    _print_daily_update_summary(
        manifest=manifest,
        report_path=report_path,
        experiment_dir=effective_experiment_dir,
        registry_rows=registry_rows,
        migration_results=migration_results,
        valuation_rows=valuation_rows,
        refresh_data=refresh_data,
        refresh_macro=refresh_macro,
        refresh_news=refresh_news,
    )
    if migration_results:
        _print_migration_table("Daily Warehouse Refresh", migration_results)


@app.command("list-snapshots")
def list_snapshots_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    snapshots = run_store.list_snapshots(limit=limit)
    if snapshots.empty:
        console.print("No completed snapshots found.")
        return
    table = Table(title="Completed Snapshots")
    for column in [
        "created_at_utc",
        "run_id",
        "market_date",
        "risk_status",
        "recommended_action",
        "risk_budget_multiplier",
        "price_columns",
        "macro_columns",
    ]:
        table.add_column(column)
    for _, row in snapshots.iterrows():
        table.add_row(
            str(row["created_at_utc"]),
            str(row["run_id"]),
            str(row["market_date"]),
            str(row["risk_status"]),
            str(row["recommended_action"]),
            f"{float(row['risk_budget_multiplier']):.2f}",
            f"{int(row['price_columns']):,}",
            f"{int(row['macro_columns']):,}",
        )
    console.print(table)


@app.command("prune-snapshots")
def prune_snapshots_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    keep_latest: Annotated[
        int,
        typer.Option("--keep-latest"),
    ] = DEFAULT_SNAPSHOT_RETENTION_KEEP_LATEST,
    keep_per_market_date: Annotated[
        int,
        typer.Option("--keep-per-market-date"),
    ] = DEFAULT_SNAPSHOT_RETENTION_KEEP_PER_MARKET_DATE,
    keep_recent_market_days: Annotated[
        int,
        typer.Option(
            "--keep-recent-market-days",
            help="Keep this many recent market dates at daily granularity.",
        ),
    ] = DEFAULT_SNAPSHOT_RETENTION_KEEP_RECENT_MARKET_DAYS,
    keep_weekly_older: Annotated[
        int,
        typer.Option(
            "--keep-weekly-older",
            help="Keep this many snapshots per older weekly bucket.",
        ),
    ] = DEFAULT_SNAPSHOT_RETENTION_KEEP_WEEKLY_OLDER,
    weekly_frequency: Annotated[
        str,
        typer.Option(
            "--weekly-frequency",
            help="Weekly retention bucket frequency for older snapshots.",
        ),
    ] = DEFAULT_SNAPSHOT_RETENTION_WEEKLY_FREQUENCY,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help="Delete snapshot artifacts and manifest rows. Defaults to dry-run.",
        ),
    ] = False,
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    candidates = run_store.prune_snapshots(
        keep_latest=keep_latest,
        keep_per_market_date=keep_per_market_date,
        keep_recent_market_days=keep_recent_market_days,
        keep_weekly_older=keep_weekly_older,
        weekly_frequency=weekly_frequency,
        apply=apply,
    )
    mode = "Applied" if apply else "Dry run"
    if candidates.empty:
        console.print(
            f"{mode}: no snapshots fall outside the retention policy "
            f"(keep_latest={keep_latest}, keep_per_market_date={keep_per_market_date}, "
            f"keep_recent_market_days={keep_recent_market_days}, "
            f"keep_weekly_older={keep_weekly_older}, weekly_frequency={weekly_frequency})."
        )
        return

    total_bytes = int(candidates["artifact_size_bytes"].sum())
    total_mb = total_bytes / (1024 * 1024)
    console.print(
        f"{mode}: {len(candidates):,} snapshot(s), {total_mb:,.1f} MB outside "
        f"retention policy (keep_latest={keep_latest}, "
        f"keep_per_market_date={keep_per_market_date}, "
        f"keep_recent_market_days={keep_recent_market_days}, "
        f"keep_weekly_older={keep_weekly_older}, weekly_frequency={weekly_frequency})."
    )
    if not apply:
        console.print("Re-run with --apply to delete these artifacts and manifest rows.")

    table = Table(title="Snapshot Prune Candidates" if not apply else "Pruned Snapshots")
    for column in [
        "created_at_utc",
        "run_id",
        "market_date",
        "risk_status",
        "artifact_size_mb",
        "pruned",
    ]:
        table.add_column(column)
    for _, row in candidates.iterrows():
        artifact_size_mb = float(row["artifact_size_bytes"]) / (1024 * 1024)
        table.add_row(
            str(row["created_at_utc"]),
            str(row["run_id"]),
            str(row["market_date"]),
            str(row["risk_status"]),
            f"{artifact_size_mb:,.1f}",
            str(bool(row["pruned"])),
        )
    console.print(table)


@app.command("backfill-snapshots")
def backfill_snapshots_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    events: Annotated[Path, typer.Option("--events")] = DEFAULT_EVENTS_PATH,
    macro: Annotated[Path, typer.Option("--macro")] = DEFAULT_MACRO_PATH,
    news: Annotated[Path, typer.Option("--news")] = DEFAULT_NEWS_PATH,
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
    refresh_macro: Annotated[bool, typer.Option("--refresh-macro")] = False,
    refresh_news: Annotated[bool, typer.Option("--refresh-news")] = False,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    start_date: Annotated[str | None, typer.Option("--start-date")] = None,
    end_date: Annotated[str | None, typer.Option("--end-date")] = None,
    years: Annotated[int, typer.Option("--years")] = DEFAULT_SNAPSHOT_BACKFILL_YEARS,
    daily_tail_days: Annotated[
        int,
        typer.Option("--daily-tail-days", help="Keep all available market dates in this tail."),
    ] = DEFAULT_SNAPSHOT_BACKFILL_DAILY_TAIL_DAYS,
    weekly_frequency: Annotated[
        str,
        typer.Option(
            "--weekly-frequency",
            help="Weekly bucket for older snapshots; the latest available date per bucket is used.",
        ),
    ] = DEFAULT_SNAPSHOT_RETENTION_WEEKLY_FREQUENCY,
    market_close_utc_hour: Annotated[
        int,
        typer.Option("--market-close-utc-hour"),
    ] = DEFAULT_SNAPSHOT_BACKFILL_MARKET_CLOSE_UTC_HOUR,
    max_snapshots: Annotated[int | None, typer.Option("--max-snapshots")] = None,
    purge_existing: Annotated[
        bool,
        typer.Option(
            "--purge-existing/--keep-existing",
            help="Delete existing snapshot artifacts and manifest rows before rebuilding.",
        ),
    ] = False,
    plan_only: Annotated[
        bool,
        typer.Option("--plan-only/--run", help="Print the rebuild schedule without writing."),
    ] = False,
) -> None:
    """Rebuild historical snapshots with daily recent and weekly older retention."""

    bot_config = load_config(config)
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    prices = load_or_fetch_yahoo_prices(
        configured_tickers(bot_config),
        start=bot_config.data.start,
        end=bot_config.data.end,
        cache_dir=bot_config.data.cache_dir,
        adjusted=bot_config.data.adjusted,
        refresh=refresh_data,
    )
    macro_catalog = load_fred_catalog(macro)
    macro_data = load_or_fetch_fred_data(
        macro_catalog,
        start=bot_config.data.start,
        end=bot_config.data.end,
        cache_dir=bot_config.data.cache_dir,
        refresh=refresh_macro,
    )
    plan = build_snapshot_backfill_plan(
        prices.index,
        start_date=start_date,
        end_date=end_date,
        years=years,
        daily_tail_days=daily_tail_days,
        weekly_frequency=weekly_frequency,
    )
    selected_dates = plan.market_dates
    if max_snapshots is not None:
        if max_snapshots <= 0:
            msg = "max_snapshots must be positive when provided."
            raise typer.BadParameter(msg)
        selected_dates = selected_dates[-max_snapshots:]

    _print_snapshot_backfill_plan(
        plan,
        selected_dates=selected_dates,
        purge_existing=purge_existing,
        plan_only=plan_only,
    )
    if plan_only:
        return

    purged = run_store.purge_snapshots(apply=True) if purge_existing else pd.DataFrame()
    if purge_existing:
        console.print(_snapshot_delete_summary("Purged existing snapshots", purged))

    saved_manifests: list[SnapshotManifest] = []
    total = len(selected_dates)
    for index, market_date in enumerate(selected_dates, start=1):
        market_date_text = str(market_date.date())
        historical_config = bot_config.model_copy(
            update={"data": bot_config.data.model_copy(update={"end": market_date_text})}
        )
        created_at_utc = snapshot_created_at_for_market_date(
            market_date,
            market_close_utc_hour=market_close_utc_hour,
        )
        historical_prices = prices.loc[:market_date_text].dropna(how="all")
        historical_macro = macro_data.loc[:market_date_text].dropna(how="all")
        baseline_run = run_configured_baselines_from_frames(
            historical_config,
            prices=historical_prices,
            macro_data=historical_macro,
            macro_catalog=macro_catalog,
            refresh_news=refresh_news,
            event_config_path=events,
            news_config_path=news,
            as_of=created_at_utc,
        )
        manifest = run_store.save_snapshot(
            baseline_run,
            config_path=config,
            events_path=events,
            macro_path=macro,
            news_path=news,
            refresh_data=False,
            refresh_macro=refresh_macro,
            refresh_news=refresh_news,
            created_at_utc=created_at_utc,
            auto_prune=False,
        )
        saved_manifests.append(manifest)
        console.print(
            f"[{index:,}/{total:,}] saved {manifest.market_date} "
            f"{manifest.risk_status} {manifest.run_id}"
        )

    recent_market_days = _selected_daily_market_day_count(
        selected_dates,
        daily_cutoff_date=plan.daily_cutoff_date,
    )
    final_prune = run_store.prune_snapshots(
        keep_latest=0,
        keep_per_market_date=DEFAULT_SNAPSHOT_RETENTION_KEEP_PER_MARKET_DATE,
        keep_recent_market_days=recent_market_days,
        keep_weekly_older=DEFAULT_SNAPSHOT_RETENTION_KEEP_WEEKLY_OLDER,
        weekly_frequency=weekly_frequency,
        apply=True,
    )
    console.print(_snapshot_delete_summary("Final retention prune", final_prune))
    console.print(
        f"Backfill complete: saved {len(saved_manifests):,} snapshot(s) from "
        f"{saved_manifests[0].market_date if saved_manifests else 'n/a'} to "
        f"{saved_manifests[-1].market_date if saved_manifests else 'n/a'}."
    )


@app.command("generate-prebreak-snapshots")
def generate_prebreak_snapshots_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    events: Annotated[Path, typer.Option("--events")] = DEFAULT_EVENTS_PATH,
    macro: Annotated[Path, typer.Option("--macro")] = DEFAULT_MACRO_PATH,
    news: Annotated[Path, typer.Option("--news")] = DEFAULT_NEWS_PATH,
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
    refresh_macro: Annotated[bool, typer.Option("--refresh-macro")] = False,
    refresh_news: Annotated[bool, typer.Option("--refresh-news")] = False,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_PREBREAK_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[
        Path, typer.Option("--artifact-dir")
    ] = DEFAULT_PREBREAK_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[
        Path, typer.Option("--job-log-dir")
    ] = DEFAULT_PREBREAK_RUN_STORE_JOB_LOG_DIR,
    lookback_days: Annotated[
        int,
        typer.Option("--lookback-days", help="Calendar days before each break date."),
    ] = DEFAULT_PREBREAK_LOOKBACK_DAYS,
    postbreak_days: Annotated[
        int,
        typer.Option("--postbreak-days", help="Calendar days after each break date."),
    ] = DEFAULT_POSTBREAK_FOLLOWTHROUGH_DAYS,
    weekly_frequency: Annotated[
        str,
        typer.Option("--weekly-frequency", help="Weekly bucket frequency for event-window dates."),
    ] = DEFAULT_PREBREAK_WEEKLY_FREQUENCY,
    market_close_utc_hour: Annotated[
        int,
        typer.Option("--market-close-utc-hour"),
    ] = DEFAULT_SNAPSHOT_BACKFILL_MARKET_CLOSE_UTC_HOUR,
    max_snapshots: Annotated[int | None, typer.Option("--max-snapshots")] = None,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--rebuild-existing",
            help="Skip market dates already present in the snapshot store.",
        ),
    ] = True,
    plan_only: Annotated[
        bool,
        typer.Option("--plan-only/--run", help="Print the pre-break schedule without writing."),
    ] = False,
) -> None:
    """Generate weekly snapshots around known bubble-break windows."""

    bot_config = load_config(config)
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    prices = load_or_fetch_yahoo_prices(
        configured_tickers(bot_config),
        start=bot_config.data.start,
        end=bot_config.data.end,
        cache_dir=bot_config.data.cache_dir,
        adjusted=bot_config.data.adjusted,
        refresh=refresh_data,
    )
    macro_catalog = load_fred_catalog(macro)
    macro_data = load_or_fetch_fred_data(
        macro_catalog,
        start=bot_config.data.start,
        end=bot_config.data.end,
        cache_dir=bot_config.data.cache_dir,
        refresh=refresh_macro,
    )
    plan = build_prebreak_snapshot_plan(
        prices.index,
        lookback_days=lookback_days,
        postbreak_days=postbreak_days,
        weekly_frequency=weekly_frequency,
    )
    if plan.empty:
        console.print("No bubble-break market dates were available.")
        return
    existing_market_dates = set()
    if skip_existing:
        existing = run_store.list_snapshots(limit=100_000)
        if not existing.empty:
            existing_market_dates = set(existing["market_date"].astype(str))
        plan = plan[~plan["market_date"].astype(str).isin(existing_market_dates)].copy()
    if max_snapshots is not None:
        if max_snapshots <= 0:
            msg = "max_snapshots must be positive when provided."
            raise typer.BadParameter(msg)
        plan = plan.tail(max_snapshots).copy()
    _print_prebreak_snapshot_plan(
        plan,
        lookback_days=lookback_days,
        postbreak_days=postbreak_days,
        weekly_frequency=weekly_frequency,
        skipped_existing=len(existing_market_dates) if skip_existing else 0,
        plan_only=plan_only,
    )
    if plan_only or plan.empty:
        return

    saved_manifests: list[SnapshotManifest] = []
    total = len(plan)
    for index, row in enumerate(plan.itertuples(index=False), start=1):
        market_date = pd.Timestamp(row.market_date)
        market_date_text = str(market_date.date())
        historical_config = bot_config.model_copy(
            update={"data": bot_config.data.model_copy(update={"end": market_date_text})}
        )
        created_at_utc = snapshot_created_at_for_market_date(
            market_date,
            market_close_utc_hour=market_close_utc_hour,
        )
        historical_prices = prices.loc[:market_date_text].dropna(how="all")
        historical_macro = macro_data.loc[:market_date_text].dropna(how="all")
        baseline_run = run_configured_baselines_from_frames(
            historical_config,
            prices=historical_prices,
            macro_data=historical_macro,
            macro_catalog=macro_catalog,
            refresh_news=refresh_news,
            event_config_path=events,
            news_config_path=news,
            as_of=created_at_utc,
        )
        manifest = run_store.save_snapshot(
            baseline_run,
            config_path=config,
            events_path=events,
            macro_path=macro,
            news_path=news,
            refresh_data=False,
            refresh_macro=refresh_macro,
            refresh_news=refresh_news,
            created_at_utc=created_at_utc,
            auto_prune=False,
        )
        saved_manifests.append(manifest)
        console.print(
            f"[{index:,}/{total:,}] saved {manifest.market_date} "
            f"{row.event_name} {manifest.risk_status} {manifest.run_id}"
        )
    console.print(f"Bubble-break snapshot generation complete: saved {len(saved_manifests):,}.")


@app.command("analyze-prebreak-hindsight")
def analyze_prebreak_hindsight_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_PREBREAK_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[
        Path, typer.Option("--artifact-dir")
    ] = DEFAULT_PREBREAK_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[
        Path, typer.Option("--job-log-dir")
    ] = DEFAULT_PREBREAK_RUN_STORE_JOB_LOG_DIR,
    reference_store: Annotated[
        Path,
        typer.Option(
            "--reference-store",
            help="Snapshot store used for current prices, current readout, and reference controls.",
        ),
    ] = DEFAULT_RUN_STORE_DB_PATH,
    reference_artifact_dir: Annotated[
        Path,
        typer.Option("--reference-artifact-dir"),
    ] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    reference_job_log_dir: Annotated[
        Path,
        typer.Option("--reference-job-log-dir"),
    ] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    include_current_snapshot: Annotated[
        bool,
        typer.Option("--include-current-snapshot/--skip-current-snapshot"),
    ] = True,
    include_reference_snapshots: Annotated[
        bool,
        typer.Option(
            "--include-reference-snapshots/--skip-reference-snapshots",
            help="Include ordinary snapshots from the reference store as non-event controls.",
        ),
    ] = True,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_PREBREAK_OUTPUT_DIR,
    lookback_days: Annotated[int, typer.Option("--lookback-days")] = DEFAULT_PREBREAK_LOOKBACK_DAYS,
    postbreak_days: Annotated[
        int,
        typer.Option("--postbreak-days", help="Calendar days after each break date."),
    ] = DEFAULT_POSTBREAK_FOLLOWTHROUGH_DAYS,
    horizon_days: Annotated[int, typer.Option("--horizon-days")] = DEFAULT_PREBREAK_HORIZON_DAYS,
    severe_drawdown_threshold: Annotated[
        float,
        typer.Option("--severe-drawdown-threshold"),
    ] = -0.10,
    major_drawdown_threshold: Annotated[
        float,
        typer.Option("--major-drawdown-threshold"),
    ] = -0.15,
) -> None:
    """Rank saved snapshot signals by hindsight 3m drawdown predictiveness."""

    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    reference_run_store = (
        RunStore(
            reference_store,
            artifact_dir=reference_artifact_dir,
            job_log_dir=reference_job_log_dir,
        )
        if include_current_snapshot or include_reference_snapshots
        else None
    )
    result = analyze_prebreak_hindsight(
        run_store,
        reference_run_store=reference_run_store,
        include_reference_snapshots=include_reference_snapshots,
        include_current_snapshot=include_current_snapshot,
        lookback_days=lookback_days,
        postbreak_days=postbreak_days,
        horizon_days=horizon_days,
        severe_drawdown_threshold=severe_drawdown_threshold,
        major_drawdown_threshold=major_drawdown_threshold,
    )
    write_prebreak_hindsight_outputs(result, output_dir=output_dir)
    console.print(
        f"Wrote pre-break hindsight analysis to {output_dir} "
        f"({len(result.snapshot_signals):,} snapshots, "
        f"{len(result.signal_rankings):,} ranked signals)."
    )
    if not result.signal_rankings.empty:
        table = Table(title="Top Hindsight Signals")
        for column in [
            "signal",
            "predictive_score",
            "spearman_to_break_severity",
            "risk_direction",
            "latest_value",
        ]:
            table.add_column(column)
        for _, row in result.signal_rankings.head(12).iterrows():
            table.add_row(
                str(row["signal"]),
                _format_optional_decimal(row["predictive_score"]),
                _format_optional_decimal(row["spearman_to_break_severity"]),
                str(row["risk_direction"]),
                _format_optional_decimal(row["latest_value"]),
            )
        console.print(table)


@app.command("list-snapshot-jobs")
def list_snapshot_jobs_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    jobs = run_store.list_jobs(limit=limit)
    if jobs.empty:
        console.print("No snapshot jobs found.")
        return
    table = Table(title="Snapshot Jobs")
    for column in [
        "created_at_utc",
        "job_id",
        "status",
        "run_id",
        "completed_at_utc",
        "log_path",
        "error_message",
    ]:
        table.add_column(column)
    for _, row in jobs.iterrows():
        table.add_row(
            str(row["created_at_utc"]),
            str(row["job_id"]),
            str(row["status"]),
            str(row["run_id"]),
            str(row["completed_at_utc"]),
            str(row["log_path"]),
            str(row["error_message"]),
        )
    console.print(table)


@app.command("run-dashboard")
def run_dashboard_cmd(
    app_path: Annotated[Path, typer.Option("--app-path")] = DEFAULT_DASHBOARD_V2_APP_PATH,
    port: Annotated[int, typer.Option("--port")] = DEFAULT_DASHBOARD_PORT,
    pid_path: Annotated[Path, typer.Option("--pid-path")] = DEFAULT_DASHBOARD_PID_PATH,
    log_path: Annotated[Path, typer.Option("--log-path")] = DEFAULT_DASHBOARD_LOG_PATH,
    file_watcher_type: Annotated[
        str,
        typer.Option(
            "--file-watcher-type",
            help="Streamlit file watcher mode. 'none' avoids common local shutdown hangs.",
        ),
    ] = DEFAULT_STREAMLIT_FILE_WATCHER_TYPE,
    stop_existing: Annotated[
        bool,
        typer.Option("--stop-existing/--keep-existing"),
    ] = False,
) -> None:
    """Start the primary Dashboard V2 as a managed background process."""

    if stop_existing:
        _stop_dashboard_from_pid_file(
            pid_path,
            port=port,
            timeout_seconds=5.0,
            force=True,
        )
    existing_pid = _read_pid_file(pid_path)
    if existing_pid is not None and _process_exists(existing_pid):
        console.print(
            f"Dashboard appears to already be running as PID {existing_pid}. "
            "Use `poetry run trade-bot run-dashboard --stop-existing` to restart it."
        )
        return
    port_pids = _listening_pids_on_port(port)
    if port_pids:
        console.print(
            f"Dashboard port {port} appears to already be in use by PID(s) "
            f"{', '.join(str(pid) for pid in port_pids)}. "
            "Use `poetry run trade-bot run-dashboard --stop-existing` to restart it."
        )
        return

    pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        file_watcher_type,
    ]
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    )
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    console.print(f"Dashboard started on http://localhost:{port} as PID {process.pid}.")
    console.print(f"Log: {log_path}")
    console.print("Stop it with: poetry run trade-bot stop-dashboard")
    console.print("Restart it with: poetry run trade-bot run-dashboard --stop-existing")


@app.command("run-dashboard-v2")
def run_dashboard_v2_cmd(
    app_path: Annotated[Path, typer.Option("--app-path")] = DEFAULT_DASHBOARD_V2_APP_PATH,
    port: Annotated[int, typer.Option("--port")] = 8502,
    pid_path: Annotated[Path, typer.Option("--pid-path")] = DEFAULT_DASHBOARD_V2_PID_PATH,
    log_path: Annotated[Path, typer.Option("--log-path")] = DEFAULT_DASHBOARD_V2_LOG_PATH,
    file_watcher_type: Annotated[
        str,
        typer.Option(
            "--file-watcher-type",
            help="Streamlit file watcher mode. 'none' avoids common local shutdown hangs.",
        ),
    ] = DEFAULT_STREAMLIT_FILE_WATCHER_TYPE,
    stop_existing: Annotated[
        bool,
        typer.Option("--stop-existing/--keep-existing"),
    ] = False,
) -> None:
    """Compatibility alias for starting Dashboard V2 on its review port."""

    run_dashboard_cmd(
        app_path=app_path,
        port=port,
        pid_path=pid_path,
        log_path=log_path,
        file_watcher_type=file_watcher_type,
        stop_existing=stop_existing,
    )


@app.command("run-dashboard-v1")
def run_dashboard_v1_cmd(
    app_path: Annotated[Path, typer.Option("--app-path")] = DEFAULT_DASHBOARD_APP_PATH,
    port: Annotated[int, typer.Option("--port")] = 8503,
    pid_path: Annotated[Path, typer.Option("--pid-path")] = Path("reports/streamlit-v1.pid"),
    log_path: Annotated[Path, typer.Option("--log-path")] = Path("reports/streamlit-v1.log"),
    file_watcher_type: Annotated[
        str,
        typer.Option(
            "--file-watcher-type",
            help="Streamlit file watcher mode. 'none' avoids common local shutdown hangs.",
        ),
    ] = DEFAULT_STREAMLIT_FILE_WATCHER_TYPE,
    stop_existing: Annotated[
        bool,
        typer.Option("--stop-existing/--keep-existing"),
    ] = False,
) -> None:
    """Start the archived Dashboard V1 fallback for comparison/debugging only."""

    console.print(
        "Dashboard V1 is archived. Use it only for fallback comparison/debugging; "
        "`poetry run trade-bot run-dashboard` is the primary V2 workbench."
    )
    run_dashboard_cmd(
        app_path=app_path,
        port=port,
        pid_path=pid_path,
        log_path=log_path,
        file_watcher_type=file_watcher_type,
        stop_existing=stop_existing,
    )


@app.command("stop-dashboard")
def stop_dashboard_cmd(
    pid_path: Annotated[Path, typer.Option("--pid-path")] = DEFAULT_DASHBOARD_PID_PATH,
    port: Annotated[
        int,
        typer.Option(
            "--port",
            help="Also stop any Streamlit process still listening on this dashboard port.",
        ),
    ] = DEFAULT_DASHBOARD_PORT,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds")] = 5.0,
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Escalate to SIGKILL if graceful stop hangs."),
    ] = True,
) -> None:
    """Stop the managed dashboard process without relying on Ctrl-C."""

    stopped = _stop_dashboard_from_pid_file(
        pid_path,
        port=port,
        timeout_seconds=timeout_seconds,
        force=force,
    )
    if not stopped:
        console.print(f"No running dashboard PID found at {pid_path}.")


@app.command("run-experiment-iteration")
def run_experiment_iteration_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    iteration: Annotated[int, typer.Option("--iteration", "-i")] = 1,
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_EXPERIMENTS_DIR,
) -> None:
    bot_config = load_config(config)
    batch = run_experiment_iteration(
        bot_config,
        iteration=iteration,
        refresh_data=refresh_data,
        output_dir=output_dir,
    )

    table = Table(title=f"Experiment Iteration {iteration:02d}")
    for column in [
        "strategy",
        "role",
        "promotion_decision",
        "promotion_score",
        "growth_constrained_utility_score",
        "growth_utility_tier",
        "cagr",
        "sharpe",
        "max_drawdown",
        "calmar",
        "worst_3y_cagr",
    ]:
        table.add_column(column)

    for name, row in batch.scorecard.iterrows():
        table.add_row(
            str(name),
            str(row["role"]),
            str(row["promotion_decision"]),
            f"{row.get('promotion_score', float('nan')):.2f}",
            _format_optional_decimal(row.get("growth_constrained_utility_score")),
            str(row.get("growth_utility_tier", "")),
            f"{row['cagr']:.2%}",
            f"{row['sharpe']:.2f}",
            f"{row['max_drawdown']:.2%}",
            f"{row['calmar']:.2f}",
            f"{row['worst_3y_cagr']:.2%}",
        )
    console.print(table)
    console.print(f"Wrote experiment outputs to {Path(output_dir) / f'iteration_{iteration:02d}'}")


@app.command("run-signal-evidence")
def run_signal_evidence_cmd(
    experiment_dir: Annotated[Path, typer.Option("--experiment-dir")] = DEFAULT_EXPERIMENTS_DIR,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_SIGNAL_EVIDENCE_DIR,
) -> None:
    effective_experiment_dir = _active_experiment_dir(experiment_dir)
    scorecards = load_experiment_scorecards(effective_experiment_dir)
    if scorecards.empty:
        console.print(f"No experiment scorecards found in {effective_experiment_dir}.")
        raise typer.Exit(code=1)

    candidates = load_experiment_candidates(effective_experiment_dir)
    tagged = tag_scorecard_signal_families(scorecards, candidates)
    evidence = build_signal_family_evidence(tagged)
    marginal_tests = build_signal_family_marginal_tests(tagged)

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "signal_family_evidence.csv"
    tests_path = output_dir / "signal_marginal_tests.csv"
    tagged_path = output_dir / "tagged_strategy_signal_families.csv"
    evidence.to_csv(evidence_path, index=False)
    marginal_tests.to_csv(tests_path, index=False)
    tagged.to_csv(tagged_path, index=False)

    table = Table(title="Signal-Family Evidence")
    for column in [
        "signal",
        "label",
        "paired",
        "candidates",
        "score",
        "median dCAGR",
        "median dMDD",
        "recommendation",
    ]:
        table.add_column(column)
    for _, row in evidence.head(12).iterrows():
        table.add_row(
            str(row["signal_label"]),
            str(row["evidence_label"]),
            str(int(row["paired_tests"])),
            str(int(row["candidate_count"])),
            _format_optional_decimal(row.get("net_evidence_score")),
            _format_optional_percent(row.get("median_delta_cagr")),
            _format_optional_percent(row.get("median_delta_max_drawdown")),
            str(row["recommendation"])[:64],
        )
    console.print(table)
    console.print(
        f"Wrote signal evidence from {effective_experiment_dir} to {output_dir} "
        f"({len(evidence):,} families, {len(marginal_tests):,} paired tests)."
    )


@app.command("run-ml-diagnostics")
def run_ml_diagnostics_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_ML_DIAGNOSTICS_DIR,
    profile: Annotated[Literal["standard", "research"], typer.Option("--profile")] = "standard",
    step_days: Annotated[int | None, typer.Option("--step-days")] = None,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    job_id: Annotated[str | None, typer.Option("--job-id")] = None,
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    if job_id:
        run_store.mark_job_running(job_id)
    try:
        bot_config = load_config(config)
        prices = load_or_fetch_yahoo_prices(
            configured_tickers(bot_config),
            start=bot_config.data.start,
            end=bot_config.data.end,
            cache_dir=bot_config.data.cache_dir,
            adjusted=bot_config.data.adjusted,
            refresh=refresh_data,
        )
        run = run_ml_diagnostics(
            prices,
            output_dir=output_dir,
            profile=profile,
            step_days=step_days,
        )
        if job_id:
            run_store.mark_job_completed(job_id, str(run.output_dir))
    except Exception as error:
        if job_id:
            run_store.mark_job_failed(job_id, str(error))
        raise
    table = Table(title=f"ML Diagnostics ({profile})")
    for column in [
        "task",
        "model",
        "utility_score",
        "balanced_accuracy",
        "brier_score",
        "calibration_error",
        "positive_recall",
    ]:
        table.add_column(column)
    for _, row in run.metrics.head(18).iterrows():
        table.add_row(
            str(row["task"]),
            str(row["model"]),
            f"{float(row['utility_score']):.3f}",
            f"{float(row['balanced_accuracy']):.3f}",
            f"{float(row['brier_score']):.3f}",
            f"{float(row['calibration_error']):.3f}",
            _format_optional_decimal(row.get("positive_recall")),
        )
    console.print(table)
    console.print(f"Wrote ML diagnostics to {run.output_dir}")


@app.command("run-entry-date-analysis")
def run_entry_date_analysis_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("reports/entry_date_analysis"),
    start_frequency: Annotated[str, typer.Option("--start-frequency")] = "M",
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    snapshot_payload = run_store.load_latest_snapshot(require_matching_config=False)
    if snapshot_payload is None:
        console.print("No completed snapshots found. Build a snapshot before entry-date analysis.")
        return
    baseline_run, manifest = snapshot_payload
    analysis = build_entry_date_analysis(
        baseline_run.results,
        start_frequency=start_frequency,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    window_path = output_dir / "entry_windows.csv"
    summary_path = output_dir / "entry_summary.csv"
    analysis.windows.to_csv(window_path, index=False)
    analysis.summary.to_csv(summary_path, index=False)

    console.print(
        f"Wrote entry-date analysis from snapshot {manifest.run_id} "
        f"({manifest.market_date}) to {output_dir}."
    )
    if analysis.summary.empty:
        console.print("No entry-date windows were available.")
        return
    preview = analysis.summary[
        analysis.summary["strategy"].isin(
            [
                "absolute_momentum_spy",
                "vol_target_dual_momentum",
                "dual_momentum_core",
                "drawdown_managed_dual_momentum",
                "buy_hold_spy",
                "buy_hold_qqq",
            ]
        )
    ].copy()
    preview = preview[preview["horizon"].isin(["3m", "1y", "3y", "5y"])]
    table = Table(title="Entry-Date Sensitivity Preview")
    for column in [
        "strategy",
        "benchmark",
        "horizon",
        "windows",
        "beat_rate",
        "median_excess_return",
        "worst_excess_return",
        "median_max_drawdown",
    ]:
        table.add_column(column)
    for _, row in preview.head(40).iterrows():
        table.add_row(
            str(row["strategy"]),
            str(row["benchmark"]),
            str(row["horizon"]),
            str(int(row["windows"])),
            _format_optional_percent(row["beat_rate"]),
            _format_optional_percent(row["median_excess_return"]),
            _format_optional_percent(row["worst_excess_return"]),
            _format_optional_percent(row["median_max_drawdown"]),
        )
    console.print(table)


@app.command("export-scenario-history")
def export_scenario_history_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[
        Path,
        typer.Option("--artifact-dir"),
    ] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[
        Path,
        typer.Option("--job-log-dir"),
    ] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    output: Annotated[
        Path,
        typer.Option("--output"),
    ] = Path("reports/simulation_validation/scenario_history.csv"),
    limit: Annotated[int, typer.Option("--limit")] = 250,
) -> None:
    """Export date-stamped scenario probabilities from saved snapshots."""

    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    history = scenario_history_from_snapshots(run_store, limit=limit)
    output_path = write_scenario_history(history, output)
    if history.empty:
        console.print(
            f"No scenario history rows were available; wrote empty file to {output_path}."
        )
        return
    market_dates = history["market_date"].dropna().nunique()
    horizons = ",".join(sorted(history["horizon"].dropna().astype(str).unique()))
    console.print(
        f"Wrote {len(history):,} scenario-history rows across {market_dates:,} "
        f"market dates to {output_path}. Horizons: {horizons}."
    )


@app.command("reconstruct-scenario-history")
def reconstruct_scenario_history_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[
        Path,
        typer.Option("--artifact-dir"),
    ] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[
        Path,
        typer.Option("--job-log-dir"),
    ] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    output: Annotated[
        Path,
        typer.Option("--output"),
    ] = Path("reports/simulation_validation/reconstructed_scenario_history.csv"),
    origin_frequency: Annotated[
        str,
        typer.Option("--origin-frequency"),
    ] = DEFAULT_FORWARD_SIMULATION_VALIDATION_ORIGIN_FREQUENCY,
    min_train_days: Annotated[
        int,
        typer.Option("--min-train-days"),
    ] = DEFAULT_FORWARD_SIMULATION_VALIDATION_MIN_TRAIN_DAYS,
    start_date: Annotated[str | None, typer.Option("--start-date")] = None,
    end_date: Annotated[str | None, typer.Option("--end-date")] = None,
) -> None:
    """Rebuild scenario history by recomputing price-derived state at past origins."""

    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    snapshot_payload = run_store.load_latest_snapshot(require_matching_config=False)
    if snapshot_payload is None:
        console.print("No completed snapshots found. Build a snapshot before reconstruction.")
        raise typer.Exit(code=1)
    baseline_run, manifest = snapshot_payload
    prices = getattr(baseline_run, "prices", pd.DataFrame())
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        console.print(f"Snapshot {manifest.run_id} does not include historical prices.")
        raise typer.Exit(code=1)
    history = reconstruct_scenario_history_from_prices(
        prices,
        origin_frequency=origin_frequency,
        min_train_days=min_train_days,
        start_date=start_date,
        end_date=end_date,
    )
    output_path = write_scenario_history(history, output)
    if history.empty:
        console.print(f"No reconstructed scenario rows were available; wrote {output_path}.")
        return
    market_dates = history["market_date"].dropna().nunique()
    horizons = ",".join(sorted(history["horizon"].dropna().astype(str).unique()))
    console.print(
        f"Reconstructed {len(history):,} scenario-history rows across {market_dates:,} "
        f"market dates from snapshot {manifest.run_id}; wrote {output_path}. "
        f"Horizons: {horizons}."
    )


@app.command("validate-simulation-engine")
def validate_simulation_engine_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path(
        "reports/simulation_validation"
    ),
    strategy: Annotated[str, typer.Option("--strategy")] = "drawdown_managed_dual_momentum",
    reference_strategies: Annotated[
        str,
        typer.Option(
            "--reference-strategies",
            help="Comma-separated strategies to include in rank validation.",
        ),
    ] = DEFAULT_SIMULATION_VALIDATION_REFERENCE_STRATEGIES,
    horizons: Annotated[
        str,
        typer.Option(
            "--horizons",
            help="Comma-separated default labels or custom label=trading_days entries.",
        ),
    ] = ",".join(DEFAULT_FORWARD_SIMULATION_VALIDATION_HORIZONS),
    origin_frequency: Annotated[
        str,
        typer.Option("--origin-frequency"),
    ] = DEFAULT_FORWARD_SIMULATION_VALIDATION_ORIGIN_FREQUENCY,
    min_train_days: Annotated[
        int,
        typer.Option("--min-train-days"),
    ] = DEFAULT_FORWARD_SIMULATION_VALIDATION_MIN_TRAIN_DAYS,
    paths: Annotated[int, typer.Option("--paths")] = DEFAULT_FORWARD_SIMULATION_PATHS,
    block_days: Annotated[
        int, typer.Option("--block-days")
    ] = DEFAULT_FORWARD_SIMULATION_BLOCK_DAYS,
    interval_low: Annotated[
        float,
        typer.Option(
            "--interval-low",
            min=0.0,
            max=0.49,
            help="Lower simulated return quantile used for rolling-origin calibration.",
        ),
    ] = DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_LOW,
    interval_high: Annotated[
        float,
        typer.Option(
            "--interval-high",
            min=0.51,
            max=1.0,
            help="Upper simulated return quantile used for rolling-origin calibration.",
        ),
    ] = DEFAULT_FORWARD_SIMULATION_VALIDATION_INTERVAL_HIGH,
    scenario_history: Annotated[
        Path | None,
        typer.Option(
            "--scenario-history",
            help="Optional date-stamped CSV or parquet scenario probabilities.",
        ),
    ] = None,
    snapshot_scenario_history_limit: Annotated[
        int,
        typer.Option(
            "--snapshot-scenario-history-limit",
            help=(
                "Use up to this many saved snapshots as date-stamped scenario history "
                "when --scenario-history is omitted. Use 0 to disable."
            ),
        ),
    ] = 0,
    ablation: Annotated[
        bool,
        typer.Option(
            "--ablation/--skip-ablation",
            help="Write a model-ablation readout comparing baseline, duration, covariate, and factor-proxy variants.",
        ),
    ] = False,
) -> None:
    """Validate forward simulation calibration against historical realized paths."""

    if interval_low >= interval_high:
        console.print("--interval-low must be lower than --interval-high.")
        raise typer.Exit(code=2)

    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    snapshot_payload = run_store.load_latest_snapshot(require_matching_config=False)
    if snapshot_payload is None:
        console.print("No completed snapshots found. Build a snapshot before validation.")
        raise typer.Exit(code=1)

    baseline_run, manifest = snapshot_payload
    selected_names = _dedupe_names([strategy, *_parse_csv_option(reference_strategies)])
    returns_by_strategy = _strategy_returns_from_results(baseline_run.results, selected_names)
    if strategy not in returns_by_strategy:
        available = ", ".join(sorted(baseline_run.results))
        console.print(f"Strategy {strategy!r} was not found in snapshot {manifest.run_id}.")
        console.print(f"Available strategies: {available}")
        raise typer.Exit(code=1)

    scenario_history_frame = _load_scenario_history(scenario_history)
    scenario_history_source = str(scenario_history or "")
    if scenario_history_frame is None and snapshot_scenario_history_limit > 0:
        scenario_history_frame = scenario_history_from_snapshots(
            run_store,
            limit=snapshot_scenario_history_limit,
        )
        if scenario_history_frame.empty:
            scenario_history_frame = None
        else:
            scenario_history_output = output_dir / "scenario_history_from_snapshots.csv"
            write_scenario_history(scenario_history_frame, scenario_history_output)
            scenario_history_source = str(scenario_history_output)
    if scenario_history_frame is None:
        console.print(
            "No date-stamped scenario history supplied; validation uses the empirical "
            "return-regime library and fallback scenario probabilities."
        )
    else:
        console.print(
            f"Loaded {len(scenario_history_frame):,} scenario-history rows from "
            f"{scenario_history_source or scenario_history}."
        )

    config = ForwardSimulationValidationConfig(
        origin_frequency=origin_frequency,
        horizons=_parse_validation_horizons(horizons),
        min_train_days=min_train_days,
        paths=paths,
        block_days=block_days,
        interval_low=interval_low,
        interval_high=interval_high,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"Running rolling-origin validation for {strategy} across "
        f"{len(config.horizons)} horizon(s), {paths} path(s) per origin."
    )
    validation = rolling_origin_simulation_backtest(
        returns_by_strategy[strategy],
        scenario_history=scenario_history_frame,
        config=config,
    )
    validation_summary = summarize_simulation_validation(validation)
    horizon_summary = summarize_simulation_validation_by_horizon(validation)
    validation_path = output_dir / f"{_slug_identifier(strategy)}_simulation_validation.csv"
    horizon_path = output_dir / f"{_slug_identifier(strategy)}_simulation_horizon_summary.csv"
    validation.to_csv(validation_path, index=False)
    horizon_summary.to_csv(horizon_path, index=False)

    console.print(
        f"Validated {strategy} from snapshot {manifest.run_id} "
        f"({manifest.market_date}); wrote {validation_path}."
    )
    _print_simulation_validation_summary(strategy, validation_summary)
    console.print(f"Wrote per-horizon validation summary to {horizon_path}.")
    _print_simulation_horizon_summary(horizon_summary)

    ablation_frame: pd.DataFrame | None = None
    ablation_path_string = ""
    if ablation:
        console.print("Running simulation model ablation variants.")
        factor_returns = _factor_returns_from_snapshot_prices(getattr(baseline_run, "prices", None))
        ablation_frame = _simulation_ablation_validation(
            returns_by_strategy[strategy],
            scenario_history=scenario_history_frame,
            factor_returns=factor_returns,
            config=config,
        )
        ablation_path = output_dir / f"{_slug_identifier(strategy)}_simulation_ablation.csv"
        ablation_frame.to_csv(ablation_path, index=False)
        ablation_path_string = str(ablation_path)
        console.print(f"Wrote simulation model ablation to {ablation_path}.")
        _print_simulation_ablation_summary(ablation_frame)

    rank_path_string = ""
    if len(returns_by_strategy) < 2:
        console.print(
            "Rank validation skipped because fewer than two selected strategies had returns."
        )
    else:
        console.print(
            f"Running strategy rank validation for {len(returns_by_strategy)} strategies."
        )
        rank_validation = rolling_origin_strategy_rank_validation(
            returns_by_strategy,
            scenario_history=scenario_history_frame,
            config=config,
        )
        rank_summary = summarize_strategy_rank_validation(rank_validation)
        rank_path = output_dir / "strategy_rank_validation.csv"
        rank_validation.to_csv(rank_path, index=False)
        rank_path_string = str(rank_path)
        console.print(f"Wrote strategy rank validation to {rank_path}.")
        _print_strategy_rank_validation_summary(rank_summary)

    validation_run_id = TradingWarehouse(store).save_simulation_validation_run(
        snapshot_run_id=manifest.run_id,
        market_date=manifest.market_date,
        strategy=strategy,
        reference_strategies=reference_strategies,
        horizons=horizons,
        origin_frequency=origin_frequency,
        min_train_days=min_train_days,
        paths=paths,
        block_days=block_days,
        interval_low=interval_low,
        interval_high=interval_high,
        scenario_history_path=str(scenario_history or ""),
        validation_output_path=str(validation_path),
        ablation_output_path=ablation_path_string,
        rank_output_path=rank_path_string,
        validation_summary=validation_summary,
        validation=validation,
        horizon_summary=horizon_summary,
        ablation_summary=ablation_frame,
    )
    console.print(f"Saved simulation validation history to DuckDB as {validation_run_id}.")


@app.command("run-cycle-tracker")
def run_cycle_tracker_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("reports/cycle_tracker"),
    horizons: Annotated[
        str,
        typer.Option(
            "--horizons",
            help="Comma-separated horizon labels or label=trading_days entries.",
        ),
    ] = "0m,1m,3m,6m,1y",
    min_train_days: Annotated[
        int,
        typer.Option(
            "--min-train-days", help="Minimum prior trading days before validation origins."
        ),
    ] = DEFAULT_PHASE_MIN_TRAIN_DAYS,
    origin_step_days: Annotated[
        int,
        typer.Option(
            "--origin-step-days",
            help="Spacing between historical validation origins. Lower is deeper but slower.",
        ),
    ] = DEFAULT_PHASE_VALIDATION_STEP_DAYS,
    candidate_tickers: Annotated[
        str,
        typer.Option(
            "--candidate-tickers",
            help="Optional comma-separated candidate tickers. Defaults to the configured broad research universe.",
        ),
    ] = "",
) -> None:
    """Build the speculative cycle tracker and conditional-winner frontier."""

    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    snapshot_payload = run_store.load_latest_snapshot(require_matching_config=False)
    if snapshot_payload is None:
        console.print(
            "No completed snapshots found. Build a snapshot before running the cycle tracker."
        )
        raise typer.Exit(code=1)
    baseline_run, manifest = snapshot_payload
    prices = getattr(baseline_run, "prices", pd.DataFrame())
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        console.print(f"Snapshot {manifest.run_id} does not include historical prices.")
        raise typer.Exit(code=1)

    parsed_horizons = tuple(days for _label, days in _parse_cycle_tracker_horizons(horizons))
    scenario_lattice = getattr(
        getattr(baseline_run, "current_state", None),
        "scenario_lattice",
        pd.DataFrame(),
    )
    result = run_cycle_tracker(
        prices=prices,
        scenario_lattice=scenario_lattice,
        output_dir=output_dir,
        candidate_tickers=tuple(_parse_csv_option(candidate_tickers)) or None,
        horizons=parsed_horizons,
        min_train_days=min_train_days,
        origin_step_days=origin_step_days,
    )
    cycle_run_id = TradingWarehouse(store).save_cycle_tracker_run(
        snapshot_run_id=manifest.run_id,
        market_date=manifest.market_date,
        output_dir=str(output_dir),
        horizons=horizons,
        min_train_days=min_train_days,
        origin_step_days=origin_step_days,
        phase_probabilities=result.phase_probabilities,
        transition_forecast=result.transition_forecast,
        evidence=result.evidence,
        candidate_scores=result.candidate_scores,
        phase_candidate_frontier=result.phase_candidate_frontier,
        validation_metrics=result.validation_metrics,
        path_validation_metrics=result.path_validation_metrics,
        path_state_history=result.path_state_history,
        path_transition_forecast=result.path_transition_forecast,
        phase_reliability=result.phase_reliability,
        path_reliability=result.path_reliability,
        crisis_playback=result.crisis_playback,
        readout=result.readout,
    )

    phase_table = Table(title="Cycle Tracker Nowcast")
    phase_table.add_column("phase")
    phase_table.add_column("probability")
    for _, row in result.phase_probabilities.sort_values("probability", ascending=False).iterrows():
        phase_table.add_row(str(row["phase"]), _format_optional_percent(row["probability"]))
    console.print(phase_table)

    forecast_table = Table(title="Scenario / Phase Frontier")
    forecast_table.add_column("horizon")
    forecast_table.add_column("dominant phase")
    forecast_table.add_column("probability")
    for horizon, group in result.transition_forecast.groupby("horizon", sort=False):
        top_row = group.sort_values("probability", ascending=False).iloc[0]
        forecast_table.add_row(
            str(horizon),
            str(top_row["phase"]),
            _format_optional_percent(top_row["probability"]),
        )
    console.print(forecast_table)

    if not result.path_transition_forecast.empty:
        path_table = Table(title="Path-Constrained Phase Frontier")
        path_table.add_column("horizon")
        path_table.add_column("dominant phase")
        path_table.add_column("probability")
        path_table.add_column("current phase")
        for horizon, group in result.path_transition_forecast.groupby("horizon", sort=False):
            top_row = group.sort_values("probability", ascending=False).iloc[0]
            path_table.add_row(
                str(horizon),
                str(top_row["phase"]),
                _format_optional_percent(top_row["probability"]),
                str(top_row.get("current_path_phase", "")),
            )
        console.print(path_table)

    candidate_table = Table(title="Conditional Winner Candidates")
    for column in ["ticker", "role", "score", "phase return", "excess vs QQQ"]:
        candidate_table.add_column(column)
    for _, row in result.candidate_scores.head(12).iterrows():
        candidate_table.add_row(
            str(row["ticker"]),
            str(row["candidate_role"]),
            _format_optional_decimal(row["candidate_score"]),
            _format_optional_percent(row.get("phase_forward_median_return")),
            _format_optional_percent(row.get("phase_median_excess_vs_qqq")),
        )
    console.print(candidate_table)
    if not result.phase_candidate_frontier.empty:
        frontier_table = Table(title="Top Scenario / Phase Winners")
        for column in ["horizon", "phase", "ticker", "role", "score", "excess vs QQQ"]:
            frontier_table.add_column(column)
        top_frontier = result.phase_candidate_frontier[
            result.phase_candidate_frontier["rank"].eq(1)
        ].head(16)
        for _, row in top_frontier.iterrows():
            frontier_table.add_row(
                str(row["horizon"]),
                str(row["phase"]),
                str(row["ticker"]),
                str(row["frontier_role"]),
                _format_optional_decimal(row["frontier_score"]),
                _format_optional_percent(row.get("median_excess_vs_qqq")),
            )
        console.print(frontier_table)
    if not result.phase_reliability.empty:
        reliability_table = Table(title="Phase Reliability")
        for column in ["horizon", "phase", "fit rate", "origins", "label"]:
            reliability_table.add_column(column)
        for _, row in result.phase_reliability.head(12).iterrows():
            reliability_table.add_row(
                str(row["horizon"]),
                str(row["dominant_phase"]),
                _format_optional_percent(row.get("phase_fit_rate")),
                str(row.get("origins", "")),
                str(row.get("reliability_label", "")),
            )
        console.print(reliability_table)
    if not result.path_reliability.empty:
        path_reliability_table = Table(title="Path Phase Reliability")
        for column in ["horizon", "path phase", "fit rate", "origins", "label"]:
            path_reliability_table.add_column(column)
        for _, row in result.path_reliability.head(12).iterrows():
            path_reliability_table.add_row(
                str(row["horizon"]),
                str(row["path_phase"]),
                _format_optional_percent(row.get("path_fit_rate")),
                str(row.get("origins", "")),
                str(row.get("reliability_label", "")),
            )
        console.print(path_reliability_table)
    console.print(f"Saved cycle tracker history to DuckDB as {cycle_run_id}.")
    console.print(f"Reports: {output_dir}")


@app.command("audit-defensive-judgement")
def audit_defensive_judgement_cmd(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("reports/defensive_signal_audit"),
    strategies: Annotated[
        str,
        typer.Option(
            "--strategies",
            help="Comma-separated strategy names. Defaults to every strategy in the latest snapshot.",
        ),
    ] = "",
    benchmarks: Annotated[
        str,
        typer.Option(
            "--benchmarks",
            help="Comma-separated risk benchmarks for false-alarm scoring.",
        ),
    ] = "SPY,QQQ",
) -> None:
    """Backfill false-alarm versus correct-defense metrics from saved backtests."""

    store = RunStore()
    baseline_run, _manifest = store.load_latest_snapshot(require_matching_config=False)
    strategy_names = [
        strategy.strip() for strategy in strategies.split(",") if strategy.strip()
    ] or None
    benchmark_tickers = [
        benchmark.strip().upper() for benchmark in benchmarks.split(",") if benchmark.strip()
    ] or ["SPY"]
    outputs = write_defensive_judgement_report(
        results=baseline_run.results,
        prices=baseline_run.prices,
        output_dir=output_dir,
        strategy_names=strategy_names,
        benchmark_tickers=benchmark_tickers,
    )
    table = Table(title="Defensive Judgement Audit")
    table.add_column("artifact")
    table.add_column("path")
    for name, path in outputs.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("audit-backtest-qc")
def audit_backtest_qc_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("reports/backtest_qc"),
    strategy: Annotated[
        str,
        typer.Option("--strategy", help="Configured strategy name to audit."),
    ] = DEFAULT_QC_STRATEGY,
    benchmarks: Annotated[
        str,
        typer.Option("--benchmarks", help="Comma-separated buy-and-hold benchmarks."),
    ] = "SPY,QQQ",
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
) -> None:
    """Run leakage, execution, universe, and parameter-sensitivity checks."""

    bot_config = load_config(config)
    prices = load_or_fetch_yahoo_prices(
        configured_tickers(bot_config),
        start=bot_config.data.start,
        end=bot_config.data.end,
        cache_dir=bot_config.data.cache_dir,
        adjusted=bot_config.data.adjusted,
        refresh=refresh_data,
    )
    benchmark_tickers = tuple(
        benchmark.strip().upper() for benchmark in benchmarks.split(",") if benchmark.strip()
    ) or ("SPY", "QQQ")
    gauntlet = run_backtest_qc_gauntlet(
        config=bot_config,
        prices=prices,
        strategy_name=strategy,
        output_dir=output_dir,
        benchmark_tickers=benchmark_tickers,
    )
    headline = gauntlet.headline.iloc[0]
    console.print(
        "[bold]Backtest QC base result:[/bold] "
        f"CAGR {_format_optional_percent(headline['cagr'])}, "
        f"max DD {_format_optional_percent(headline['max_drawdown'])}, "
        f"Sharpe {float(headline['sharpe']):.2f}."
    )
    table = Table(title="Backtest QC Artifacts")
    table.add_column("artifact")
    table.add_column("path")
    for name, path in gauntlet.artifacts.items():
        table.add_row(name, str(path))
    console.print(table)
    console.print(gauntlet.readout)


@app.command("run-leadership-diagnostics")
def run_leadership_diagnostics_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("reports/leadership_diagnostics"),
    experiment_root: Annotated[
        Path,
        typer.Option("--experiment-root"),
    ] = DEFAULT_EXPERIMENTS_DIR,
    strategies: Annotated[
        str,
        typer.Option(
            "--strategies",
            help="Comma-separated strategies to force; defaults to primary/configured plus top candidates.",
        ),
    ] = "",
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Maximum top candidate count before configured strategies."),
    ] = 5,
    router_horizons: Annotated[
        str,
        typer.Option(
            "--router-horizons",
            help="Comma-separated trading-day horizons for the walk-forward router.",
        ),
    ] = "21,63,126",
    router_step_days: Annotated[
        int,
        typer.Option(
            "--router-step-days",
            help="Spacing between walk-forward router origins. Lower is deeper but slower.",
        ),
    ] = 126,
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
) -> None:
    """Audit tech leadership dependence and run the prior-only strategy router."""

    bot_config = load_config(config)
    strategy_names = tuple(
        strategy.strip() for strategy in strategies.split(",") if strategy.strip()
    )
    tickers = leadership_candidate_tickers(
        bot_config,
        experiment_root=experiment_root,
        strategies=strategy_names,
        top_n=top_n,
    )
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=bot_config.data.start,
        end=bot_config.data.end,
        cache_dir=bot_config.data.cache_dir,
        adjusted=bot_config.data.adjusted,
        refresh=refresh_data,
    )
    horizons = tuple(
        int(value.strip()) for value in router_horizons.split(",") if value.strip()
    ) or (21, 63, 126)
    result = run_leadership_diagnostics(
        config=bot_config,
        prices=prices,
        output_dir=output_dir,
        experiment_root=experiment_root,
        strategies=strategy_names,
        top_n=top_n,
        router_horizons=horizons,
        origin_step_days=router_step_days,
    )

    table = Table(title="Leadership Diagnostics")
    for column in [
        "strategy",
        "current tech/AI",
        "avg tech/AI",
        "current mega-cap",
        "current non-tech",
    ]:
        table.add_column(column)
    for _, row in result.tech_dependence.head(12).iterrows():
        table.add_row(
            str(row["strategy"]),
            _format_optional_percent(row.get("current_tech_ai_weight")),
            _format_optional_percent(row.get("avg_tech_ai_weight")),
            _format_optional_percent(row.get("current_mega_cap_tech_weight")),
            _format_optional_percent(row.get("current_non_tech_weight")),
        )
    console.print(table)
    if not result.router_summary.empty:
        router_table = Table(title="Walk-Forward Router Summary")
        for column in [
            "horizon",
            "folds",
            "pick excess",
            "pick hit",
            "blend excess",
            "blend hit",
        ]:
            router_table.add_column(column)
        for _, row in result.router_summary.iterrows():
            router_table.add_row(
                str(int(row["horizon_days"])),
                str(int(row["folds"])),
                _format_optional_percent(row.get("selected_mean_excess_vs_benchmark")),
                _format_optional_percent(row.get("selected_hit_rate")),
                _format_optional_percent(row.get("top3_blend_mean_excess_vs_benchmark")),
                _format_optional_percent(row.get("top3_blend_hit_rate")),
            )
        console.print(router_table)
    console.print(f"Reports: {output_dir}")


@app.command("audit-backtest-pbo")
def audit_backtest_pbo_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = Path("reports/pbo_diagnostics"),
    experiment_root: Annotated[
        Path,
        typer.Option("--experiment-root"),
    ] = DEFAULT_EXPERIMENTS_DIR,
    strategies: Annotated[
        str,
        typer.Option(
            "--strategies",
            help="Comma-separated strategies to force; defaults to primary plus top candidates.",
        ),
    ] = "",
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Maximum candidate count in the PBO matrix."),
    ] = 20,
    partitions: Annotated[
        int,
        typer.Option("--partitions", help="Even CSCV block count. 8 gives 70 train/test splits."),
    ] = 8,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Selection metric: sharpe, mean_return, or total_return."),
    ] = "sharpe",
    min_observations: Annotated[
        int,
        typer.Option("--min-observations", help="Minimum return observations per candidate."),
    ] = 252,
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
) -> None:
    """Estimate Probability of Backtest Overfitting using CSCV over candidate returns."""

    bot_config = load_config(config)
    strategy_names = tuple(
        strategy.strip() for strategy in strategies.split(",") if strategy.strip()
    )
    tickers = set(configured_tickers(bot_config)) | pbo_candidate_tickers(
        bot_config,
        experiment_root=experiment_root,
        strategies=strategy_names,
        top_n=top_n,
    )
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=bot_config.data.start,
        end=bot_config.data.end,
        cache_dir=bot_config.data.cache_dir,
        adjusted=bot_config.data.adjusted,
        refresh=refresh_data,
    )
    gauntlet = run_backtest_pbo_gauntlet(
        config=bot_config,
        prices=prices,
        output_dir=output_dir,
        experiment_root=experiment_root,
        strategies=strategy_names,
        top_n=top_n,
        partitions=partitions,
        metric=metric,  # type: ignore[arg-type]
        min_observations=min_observations,
    )
    summary = gauntlet.result.summary.iloc[0]
    console.print(
        "[bold]Backtest PBO result:[/bold] "
        f"PBO {_format_optional_percent(summary.get('pbo_probability'))}, "
        f"OOS loss {_format_optional_percent(summary.get('oos_loss_probability'))}, "
        f"label {summary.get('pbo_label')}."
    )
    table = Table(title="Backtest PBO Artifacts")
    table.add_column("artifact")
    table.add_column("path")
    for name, path in gauntlet.artifacts.items():
        table.add_row(name, str(path))
    console.print(table)
    console.print(gauntlet.readout)


@app.command("migrate-warehouse")
def migrate_warehouse_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    experiment_dir: Annotated[Path, typer.Option("--experiment-dir")] = DEFAULT_EXPERIMENTS_DIR,
    journal: Annotated[Path, typer.Option("--journal")] = DEFAULT_JOURNAL_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    job_id: Annotated[str | None, typer.Option("--job-id")] = None,
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    if job_id:
        run_store.mark_job_running(job_id)
    try:
        warehouse = TradingWarehouse(store)
        experiment_results = warehouse.migrate_experiment_outputs(experiment_dir)
        journal_results = warehouse.migrate_journal_sqlite(journal)
        if job_id:
            run_store.mark_job_completed(job_id, "warehouse-migration")
    except Exception as error:
        if job_id:
            run_store.mark_job_failed(job_id, str(error))
        raise
    _print_migration_table(
        "Warehouse Migration",
        [*experiment_results, *journal_results],
    )
    console.print("[bold]Warehouse table counts[/bold]")
    console.print(warehouse.table_counts())


@app.command("audit-strategy-sources")
def audit_strategy_sources_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DEFAULT_STRATEGY_SOURCE_AUDIT_DIR,
    experiment_dir: Annotated[
        list[Path] | None,
        typer.Option(
            "--experiment-dir",
            help=(
                "Experiment roots to reconcile. Defaults to reports/experiments and "
                "data/experiments_reset_v2."
            ),
        ),
    ] = None,
    top_n: Annotated[int, typer.Option("--top-n")] = 50,
) -> None:
    roots = (
        tuple(experiment_dir)
        if experiment_dir
        else (DEFAULT_EXPERIMENTS_DIR, DEFAULT_RESET_EXPERIMENTS_DIR)
    )
    audit = write_strategy_source_audit(
        output_dir=output_dir,
        warehouse_path=store,
        experiment_roots=roots,
        top_n=top_n,
    )
    console.print(f"Wrote strategy source audit to {output_dir}.")
    if audit.full_history_top.empty:
        console.print("No full-history strategy metrics found.")
        return
    table = Table(title="Top Full-History Strategy Sources")
    for column in ["source_scope", "strategy", "cagr", "max_drawdown", "calmar", "source_path"]:
        table.add_column(column)
    for _, row in audit.full_history_top.head(10).iterrows():
        table.add_row(
            str(row.get("source_scope", "")),
            str(row.get("strategy", "")),
            _format_optional_percent(row.get("cagr")),
            _format_optional_percent(row.get("max_drawdown")),
            _format_optional_decimal(row.get("calmar")),
            str(row.get("source_path", ""))[:90],
        )
    console.print(table)
    console.print(
        f"High-CAGR metric hits: {len(audit.high_cagr_metric_hits):,}; "
        f"text/doc references: {len(audit.ambiguous_references):,}."
    )


@app.command("seed-operating-history")
def seed_operating_history_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    events: Annotated[Path, typer.Option("--events")] = DEFAULT_EVENTS_PATH,
    macro: Annotated[Path, typer.Option("--macro")] = DEFAULT_MACRO_PATH,
    news: Annotated[Path, typer.Option("--news")] = DEFAULT_NEWS_PATH,
    source: Annotated[
        Literal["latest-snapshot", "configured-baselines"],
        typer.Option(
            "--source",
            help=(
                "Use the latest saved snapshot when available, or recompute configured "
                "baselines from local inputs."
            ),
        ),
    ] = "latest-snapshot",
    start_date: Annotated[str | None, typer.Option("--start-date")] = None,
    end_date: Annotated[str | None, typer.Option("--end-date")] = None,
    frequency: Annotated[
        str,
        typer.Option(
            "--frequency",
            help="Pandas resample frequency for historical points, for example W-WED, ME, B.",
        ),
    ] = "W-WED",
    max_points: Annotated[int, typer.Option("--max-points")] = 260,
    daily_tail_market_days: Annotated[
        int,
        typer.Option(
            "--daily-tail-market-days",
            help="Append this many recent market dates at daily granularity.",
        ),
    ] = 30,
    min_history_days: Annotated[int, typer.Option("--min-history-days")] = 252,
    primary_strategy: Annotated[
        str,
        typer.Option("--primary-strategy"),
    ] = DEFAULT_OPERATING_HISTORY_PRIMARY_STRATEGY,
) -> None:
    run_store = RunStore(store)
    source_label = "configured baselines"
    snapshot_payload = None
    if source == "latest-snapshot":
        try:
            snapshot_payload = run_store.load_latest_snapshot(require_matching_config=False)
        except (FileNotFoundError, TypeError, OSError, AttributeError, ValueError) as error:
            console.print(
                f"Could not load latest snapshot; falling back to baseline recompute: {error}"
            )
    if snapshot_payload is not None:
        baseline_run, manifest = snapshot_payload
        source_label = f"latest snapshot {manifest.run_id}"
    else:
        if source == "latest-snapshot":
            console.print("No completed snapshot found; recomputing configured baselines.")
        bot_config = load_config(config)
        baseline_run = run_configured_baselines(
            bot_config,
            refresh_data=False,
            refresh_macro=False,
            refresh_news=False,
            event_config_path=events,
            macro_config_path=macro,
            news_config_path=news,
        )
    history = reconstruct_operating_history(
        baseline_run,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        max_points=max_points,
        daily_tail_market_days=daily_tail_market_days,
        min_history_days=min_history_days,
        primary_strategy=primary_strategy,
    )
    warehouse = TradingWarehouse(store)
    counts = warehouse.save_operating_history(
        metrics=history.metrics,
        components=history.components,
        scenario_drivers=history.scenario_drivers,
        driver_rotation=history.driver_rotation,
        replace_sources=(DEFAULT_OPERATING_HISTORY_SOURCE,),
    )
    table = Table(title="Seeded Operating History")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for table_name, rows in counts.items():
        table.add_row(table_name, f"{rows:,}")
    console.print(table)
    if history.metrics.empty:
        console.print(
            "No operating history rows were generated. Check start/end dates, frequency, "
            "and whether enough price history exists for the requested min-history-days."
        )
    else:
        start = str(history.metrics["market_date"].iloc[0])
        end = str(history.metrics["market_date"].iloc[-1])
        console.print(
            f"Stored reconstructed operating history from {start} to {end} using {source_label}. "
            "These rows are separate from live saved snapshots."
        )


@app.command("sync-42macro-transcripts")
def sync_42macro_transcripts_cmd(
    transcript_dir: Annotated[
        Path,
        typer.Option("--transcript-dir"),
    ] = DEFAULT_EXTERNAL_MACRO_TRANSCRIPT_DIR,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    max_videos: Annotated[int | None, typer.Option("--max-videos")] = 250,
    max_pages: Annotated[int | None, typer.Option("--max-pages")] = 25,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    channel_handle: Annotated[str, typer.Option("--channel-handle")] = DEFAULT_42MACRO_HANDLE,
    transcript_timeout_seconds: Annotated[
        int,
        typer.Option("--transcript-timeout-seconds"),
    ] = 30,
    metadata_only: Annotated[bool, typer.Option("--metadata-only")] = False,
) -> None:
    result = sync_42macro_transcripts(
        transcript_dir=transcript_dir,
        max_videos=max_videos,
        max_pages=max_pages,
        refresh=refresh,
        channel_handle=channel_handle,
        transcript_timeout_seconds=transcript_timeout_seconds,
        fetch_transcripts=not metadata_only,
    )
    warehouse = TradingWarehouse(store)
    warehouse.save_external_macro_alignment(
        videos=result.videos,
        classifications=pd.DataFrame(),
        comparisons=pd.DataFrame(),
    )
    table = Table(title="42 Macro Transcript Sync")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("catalog videos", f"{len(result.videos):,}")
    table.add_row("fetched", f"{result.fetched:,}")
    table.add_row("skipped existing", f"{result.skipped:,}")
    table.add_row("failed", f"{result.failed:,}")
    console.print(table)
    console.print(f"Manifest: {result.manifest_path}")


@app.command("import-42macro-transcripts")
def import_42macro_transcripts_cmd(
    input_dir: Annotated[
        Path,
        typer.Option("--input-dir"),
    ],
    transcript_dir: Annotated[
        Path,
        typer.Option("--transcript-dir"),
    ] = DEFAULT_EXTERNAL_MACRO_TRANSCRIPT_DIR,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    result = import_42macro_transcript_files(
        input_dir=input_dir,
        transcript_dir=transcript_dir,
        overwrite=overwrite,
    )
    warehouse = TradingWarehouse(store)
    warehouse.save_external_macro_alignment(
        videos=result.videos,
        classifications=pd.DataFrame(),
        comparisons=pd.DataFrame(),
    )
    table = Table(title="42 Macro Manual Transcript Import")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("manifest videos", f"{len(result.videos):,}")
    table.add_row("imported", f"{result.imported:,}")
    table.add_row("skipped existing", f"{result.skipped:,}")
    table.add_row("failed", f"{result.failed:,}")
    console.print(table)
    if result.failed:
        failed = result.imported_files[result.imported_files["status"].eq("failed")]
        console.print(failed[["input_path", "error"]].head(10).to_string(index=False))
    console.print(f"Manifest: {result.manifest_path}")


@app.command("prioritize-42macro-transcripts")
def prioritize_42macro_transcripts_cmd(
    transcript_dir: Annotated[
        Path,
        typer.Option("--transcript-dir"),
    ] = DEFAULT_EXTERNAL_MACRO_TRANSCRIPT_DIR,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = DEFAULT_EXTERNAL_MACRO_ALIGNMENT_DIR,
    comparison_path: Annotated[
        Path | None,
        typer.Option("--comparison-path"),
    ] = None,
    outcome_path: Annotated[
        Path | None,
        typer.Option("--outcome-path"),
    ] = None,
    top_n: Annotated[int, typer.Option("--top-n")] = 25,
) -> None:
    priority = write_missing_42macro_transcript_priority(
        transcript_dir=transcript_dir,
        output_dir=output_dir,
        comparison_path=comparison_path,
        outcome_path=outcome_path,
    )
    path = output_dir / "missing_transcript_priority.csv"
    console.print(f"Wrote {len(priority):,} missing transcript priorities: {path}")
    if not priority.empty:
        console.print(
            priority[["priority_score", "published_date", "video_id", "title", "transcript_url"]]
            .head(top_n)
            .to_string(index=False)
        )


@app.command("compare-42macro")
def compare_42macro_cmd(
    transcript_dir: Annotated[
        Path,
        typer.Option("--transcript-dir"),
    ] = DEFAULT_EXTERNAL_MACRO_TRANSCRIPT_DIR,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = DEFAULT_EXTERNAL_MACRO_ALIGNMENT_DIR,
    max_match_days: Annotated[int, typer.Option("--max-match-days")] = 10,
) -> None:
    warehouse = TradingWarehouse(store)
    result = compare_42macro_to_trade_bot(
        transcript_dir=transcript_dir,
        warehouse=warehouse,
        output_dir=output_dir,
        max_match_days=max_match_days,
    )
    _print_macro_alignment_summary(result.summary, output_dir)


@app.command("score-42macro-outcomes")
def score_42macro_outcomes_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    comparison_path: Annotated[
        Path,
        typer.Option("--comparison-path"),
    ] = DEFAULT_EXTERNAL_MACRO_ALIGNMENT_DIR / "daily_comparison.csv",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = DEFAULT_EXTERNAL_MACRO_ALIGNMENT_DIR,
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
    risk_ticker: Annotated[str, typer.Option("--risk-ticker")] = "SPY",
    defensive_ticker: Annotated[str, typer.Option("--defensive-ticker")] = "BIL",
) -> None:
    if not comparison_path.exists():
        raise typer.BadParameter(
            f"{comparison_path} does not exist. Run `trade-bot compare-42macro` first."
        )
    bot_config = load_config(config)
    prices = load_or_fetch_yahoo_prices(
        configured_tickers(bot_config),
        start=bot_config.data.start,
        end=bot_config.data.end,
        cache_dir=bot_config.data.cache_dir,
        adjusted=bot_config.data.adjusted,
        refresh=refresh_data,
    )
    comparisons = pd.read_csv(comparison_path)
    result = score_macro_tradebot_outcomes(
        comparisons=comparisons,
        prices=prices,
        output_dir=output_dir,
        risk_ticker=risk_ticker,
        defensive_ticker=defensive_ticker,
    )
    _print_macro_outcome_summary(result.summary, output_dir)


@app.command("run-42macro-daily-check")
def run_42macro_daily_check_cmd(
    transcript_dir: Annotated[
        Path,
        typer.Option("--transcript-dir"),
    ] = DEFAULT_EXTERNAL_MACRO_TRANSCRIPT_DIR,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = DEFAULT_EXTERNAL_MACRO_ALIGNMENT_DIR,
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    max_videos: Annotated[int | None, typer.Option("--max-videos")] = 80,
    max_pages: Annotated[int | None, typer.Option("--max-pages")] = 8,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    max_match_days: Annotated[int, typer.Option("--max-match-days")] = 10,
    channel_handle: Annotated[str, typer.Option("--channel-handle")] = DEFAULT_42MACRO_HANDLE,
    transcript_timeout_seconds: Annotated[
        int,
        typer.Option("--transcript-timeout-seconds"),
    ] = 30,
    metadata_only: Annotated[bool, typer.Option("--metadata-only")] = False,
    score_outcomes: Annotated[bool, typer.Option("--score-outcomes/--skip-outcomes")] = True,
) -> None:
    sync_result = sync_42macro_transcripts(
        transcript_dir=transcript_dir,
        max_videos=max_videos,
        max_pages=max_pages,
        refresh=refresh,
        channel_handle=channel_handle,
        transcript_timeout_seconds=transcript_timeout_seconds,
        fetch_transcripts=not metadata_only,
    )
    warehouse = TradingWarehouse(store)
    result = compare_42macro_to_trade_bot(
        transcript_dir=transcript_dir,
        warehouse=warehouse,
        output_dir=output_dir,
        max_match_days=max_match_days,
    )
    console.print(
        f"Synced {len(sync_result.videos):,} public 42 Macro videos "
        f"({sync_result.fetched:,} fetched, {sync_result.skipped:,} skipped)."
    )
    _print_macro_alignment_summary(result.summary, output_dir)
    if score_outcomes and not result.comparisons.empty:
        bot_config = load_config(config)
        prices = load_or_fetch_yahoo_prices(
            configured_tickers(bot_config),
            start=bot_config.data.start,
            end=bot_config.data.end,
            cache_dir=bot_config.data.cache_dir,
            adjusted=bot_config.data.adjusted,
            refresh=False,
        )
        outcome_result = score_macro_tradebot_outcomes(
            comparisons=result.comparisons,
            prices=prices,
            output_dir=output_dir,
        )
        _print_macro_outcome_summary(outcome_result.summary, output_dir)


@app.command("run-upside-capture-lab")
def run_upside_capture_lab_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir"),
    ] = DEFAULT_UPSIDE_CAPTURE_OUTPUT_DIR,
    primary_strategy: Annotated[
        str,
        typer.Option("--primary-strategy"),
    ] = "i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145",
    refresh_data: Annotated[bool, typer.Option("--refresh-data")] = False,
) -> None:
    bot_config = load_config(config)
    result = run_upside_capture_lab(
        bot_config,
        output_dir=output_dir,
        primary_strategy=primary_strategy,
        refresh_data=refresh_data,
    )
    table = Table(title="Upside Capture Lab")
    table.add_column("candidate")
    table.add_column("round", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("max DD", justify="right")
    table.add_column("run-up lift", justify="right")
    table.add_column("left-tail delta", justify="right")
    table.add_column("score", justify="right")
    for _, row in result.summary.sort_values("research_score", ascending=False).head(10).iterrows():
        table.add_row(
            str(row["candidate"]),
            str(int(row["round_id"])),
            _format_optional_percent(row.get("cagr")),
            _format_optional_percent(row.get("max_drawdown")),
            _format_optional_percent(row.get("runup_capture_lift")),
            _format_optional_percent(row.get("left_tail_loss_delta")),
            _format_optional_decimal(row.get("research_score")),
        )
    console.print(table)
    console.print(f"Reports: {output_dir}")


@app.command("seed-monitoring-windows")
def seed_monitoring_windows_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    mode: Annotated[str, typer.Option("--mode")] = "paper",
    account: Annotated[str, typer.Option("--account")] = "default_paper_account",
    capital_base: Annotated[float, typer.Option("--capital-base")] = 10_000.0,
    top_n: Annotated[int, typer.Option("--top-n")] = DEFAULT_MONITORING_TOP_N,
    start_date: Annotated[str | None, typer.Option("--start-date")] = None,
    job_id: Annotated[str | None, typer.Option("--job-id")] = None,
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    if job_id:
        run_store.mark_job_running(job_id)
    try:
        warehouse = TradingWarehouse(store)
        snapshot_payload = run_store.load_latest_snapshot(require_matching_config=False)
        if snapshot_payload is not None:
            baseline_run, manifest = snapshot_payload
            warehouse.refresh_strategy_registry_from_snapshot(
                baseline_run,
                run_id=manifest.run_id,
                market_date=manifest.market_date,
            )
        seeded = warehouse.seed_monitoring_windows_from_registry(
            mode=mode,
            account=account,
            capital_base=capital_base,
            top_n=top_n,
            start_date=start_date,
        )
        if job_id:
            run_store.mark_job_completed(job_id, "monitoring-seed")
    except Exception as error:
        if job_id:
            run_store.mark_job_failed(job_id, str(error))
        raise
    if not seeded:
        console.print("No new monitoring windows were seeded.")
        return
    table = Table(title="Seeded Monitoring Windows")
    for column in ["window_id", "strategy_id", "strategy_name", "role"]:
        table.add_column(column)
    for row in seeded:
        table.add_row(row.window_id, row.strategy_id, row.strategy_name, row.role)
    console.print(table)


@app.command("run-paper-valuation")
def run_paper_valuation_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    job_id: Annotated[str | None, typer.Option("--job-id")] = None,
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    if job_id:
        run_store.mark_job_running(job_id)
    try:
        snapshot_payload = run_store.load_latest_snapshot(require_matching_config=False)
        if snapshot_payload is None:
            console.print("No completed snapshots found. Build a snapshot before paper valuation.")
            if job_id:
                run_store.mark_job_completed(job_id, "paper-valuation-no-snapshot")
            return
        baseline_run, manifest = snapshot_payload
        warehouse = TradingWarehouse(store)
        warehouse.refresh_strategy_registry_from_snapshot(
            baseline_run,
            run_id=manifest.run_id,
            market_date=manifest.market_date,
        )
        bot_config = load_config(config)
        rows = warehouse.save_daily_valuations_from_snapshot(
            baseline_run,
            market_date=manifest.market_date,
            execution=bot_config.execution,
        )
        if job_id:
            run_store.mark_job_completed(job_id, manifest.run_id)
    except Exception as error:
        if job_id:
            run_store.mark_job_failed(job_id, str(error))
        raise
    console.print(
        f"Wrote {rows:,} paper valuation rows from snapshot {manifest.run_id} "
        f"for market date {manifest.market_date}."
    )


@app.command("reset-monitoring-start-date")
def reset_monitoring_start_date_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")] = DEFAULT_RUN_STORE_ARTIFACT_DIR,
    job_log_dir: Annotated[Path, typer.Option("--job-log-dir")] = DEFAULT_RUN_STORE_JOB_LOG_DIR,
    start_date: Annotated[str, typer.Option("--start-date")] = DEFAULT_MONITORING_COHORT_START_DATE,
    mode: Annotated[str | None, typer.Option("--mode")] = "paper",
    account: Annotated[str | None, typer.Option("--account")] = None,
    status: Annotated[str | None, typer.Option("--status")] = "active",
    value_after_reset: Annotated[
        bool,
        typer.Option("--value-after-reset/--skip-valuation"),
    ] = True,
    job_id: Annotated[str | None, typer.Option("--job-id")] = None,
) -> None:
    run_store = RunStore(store, artifact_dir=artifact_dir, job_log_dir=job_log_dir)
    if job_id:
        run_store.mark_job_running(job_id)
    valuation_rows = 0
    try:
        warehouse = TradingWarehouse(store)
        reset_rows = warehouse.reset_monitoring_start_dates(
            start_date=start_date,
            mode=mode,
            account=account,
            status=status,
            clear_valuations=True,
        )
        if value_after_reset and reset_rows:
            snapshot_payload = run_store.load_latest_snapshot(require_matching_config=False)
            if snapshot_payload is None:
                console.print("No completed snapshots found. Monitoring starts were reset only.")
            else:
                baseline_run, manifest = snapshot_payload
                warehouse.refresh_strategy_registry_from_snapshot(
                    baseline_run,
                    run_id=manifest.run_id,
                    market_date=manifest.market_date,
                )
                bot_config = load_config(config)
                valuation_rows = warehouse.save_daily_valuations_from_snapshot(
                    baseline_run,
                    market_date=manifest.market_date,
                    execution=bot_config.execution,
                )
        if job_id:
            run_store.mark_job_completed(job_id, "monitoring-start-reset")
    except Exception as error:
        if job_id:
            run_store.mark_job_failed(job_id, str(error))
        raise
    scope = f"mode={mode or 'all'}"
    if account:
        scope += f", account={account}"
    console.print(
        f"Reset {reset_rows:,} monitoring window(s) to start date {start_date} "
        f"({scope}); wrote {valuation_rows:,} valuation row(s)."
    )


@app.command("monitor-strategy")
def monitor_strategy_cmd(
    strategy_name: Annotated[str, typer.Argument(help="Strategy name or strategy id to monitor")],
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    role: Annotated[str, typer.Option("--role")] = "challenger",
    mode: Annotated[str, typer.Option("--mode")] = "paper",
    account: Annotated[str, typer.Option("--account")] = "default_paper_account",
    capital_base: Annotated[float, typer.Option("--capital-base")] = 10_000.0,
    start_date: Annotated[str | None, typer.Option("--start-date")] = None,
    demote_other_champions: Annotated[
        bool,
        typer.Option("--demote-other-champions"),
    ] = False,
) -> None:
    warehouse = TradingWarehouse(store)
    result = warehouse.monitor_strategy(
        strategy_name,
        role=role,
        mode=mode,
        account=account,
        capital_base=capital_base,
        start_date=start_date,
        demote_other_champions=demote_other_champions,
    )
    console.print(
        f"Monitoring {result.strategy_name} as {result.role} in window {result.window_id}."
    )


@app.command("update-monitoring-window")
def update_monitoring_window_cmd(
    window_id: Annotated[str, typer.Argument(help="Monitoring window id to update")],
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    role: Annotated[str | None, typer.Option("--role")] = None,
    status: Annotated[str | None, typer.Option("--status")] = None,
    capital_base: Annotated[float | None, typer.Option("--capital-base")] = None,
    start_date: Annotated[str | None, typer.Option("--start-date")] = None,
    demote_other_champions: Annotated[
        bool,
        typer.Option("--demote-other-champions"),
    ] = False,
) -> None:
    warehouse = TradingWarehouse(store)
    updated = warehouse.update_monitoring_window(
        window_id,
        role=role,
        status=status,
        capital_base=capital_base,
        start_date=start_date,
        demote_other_champions=demote_other_champions,
    )
    if not updated:
        console.print(f"No monitoring window found: {window_id}")
        raise typer.Exit(code=1)
    console.print(f"Updated monitoring window {window_id}.")


@app.command("list-monitoring-windows")
def list_monitoring_windows_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
    status: Annotated[str, typer.Option("--status")] = "active",
) -> None:
    warehouse = TradingWarehouse(store)
    windows = warehouse.list_monitoring_windows(status=None if status == "all" else status)
    if windows.empty:
        console.print("No monitoring windows found.")
        return
    table = Table(title="Monitoring Windows")
    for column in [
        "window_role",
        "mode",
        "account",
        "strategy_name",
        "status",
        "start_date",
        "capital_base",
        "notes",
    ]:
        table.add_column(column)
    for _, row in windows.iterrows():
        table.add_row(
            str(row["window_role"]),
            str(row["mode"]),
            str(row["account"]),
            str(row["strategy_name"]),
            str(row["status"]),
            str(row["start_date"]),
            f"${float(row['capital_base']):,.0f}",
            str(row["notes"])[:90],
        )
    console.print(table)


@app.command("list-champion-challenger")
def list_champion_challenger_cmd(
    store: Annotated[Path, typer.Option("--store")] = DEFAULT_RUN_STORE_DB_PATH,
) -> None:
    warehouse = TradingWarehouse(store)
    frame = warehouse.champion_challenger_frame()
    if frame.empty:
        console.print("No champion/challenger monitoring rows found.")
        return
    table = Table(title="Champion / Challenger")
    for column in [
        "window_role",
        "strategy_name",
        "forward_status",
        "cumulative_return",
        "excess_return",
        "drawdown",
        "promotion_score",
        "overfit_risk_label",
        "validation_tier",
    ]:
        table.add_column(column)
    for _, row in frame.iterrows():
        table.add_row(
            str(row.get("window_role", "")),
            str(row.get("strategy_name", "")),
            str(row.get("forward_status", "")),
            _format_optional_percent(row.get("cumulative_return")),
            _format_optional_percent(row.get("excess_return")),
            _format_optional_percent(row.get("drawdown")),
            _format_optional_decimal(row.get("promotion_score")),
            str(row.get("overfit_risk_label", "")),
            str(row.get("validation_tier", "")),
        )
    console.print(table)


def _parse_csv_option(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe_names(names: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        output.append(name)
    return output


def _parse_validation_horizons(value: str) -> tuple[tuple[str, int], ...]:
    requested = _parse_csv_option(value)
    if not requested:
        msg = "At least one horizon is required."
        raise typer.BadParameter(msg)
    horizons: list[tuple[str, int]] = []
    defaults = DEFAULT_FORWARD_SIMULATION_VALIDATION_HORIZONS
    for item in requested:
        if "=" in item:
            label, raw_days = item.split("=", 1)
            label = label.strip()
            try:
                days = int(raw_days.strip())
            except ValueError as error:
                msg = f"Invalid horizon day count: {item!r}"
                raise typer.BadParameter(msg) from error
        else:
            label = item.strip()
            if label not in defaults:
                available = ", ".join(defaults)
                msg = f"Unknown horizon {label!r}; use one of {available} or label=days."
                raise typer.BadParameter(msg)
            days = int(defaults[label])
        if not label or days <= 0:
            msg = f"Invalid horizon: {item!r}"
            raise typer.BadParameter(msg)
        horizons.append((label, days))
    return tuple(horizons)


def _parse_cycle_tracker_horizons(value: str) -> tuple[tuple[str, int], ...]:
    requested = _parse_csv_option(value)
    if not requested:
        msg = "At least one horizon is required."
        raise typer.BadParameter(msg)
    defaults = {
        "0m": 0,
        "nowcast": 0,
        "1w": 5,
        "1m": 21,
        "2m": 42,
        "3m": 63,
        "6m": 126,
        "1y": 252,
    }
    horizons: list[tuple[str, int]] = []
    for item in requested:
        if "=" in item:
            label, raw_days = item.split("=", 1)
            label = label.strip()
            try:
                days = int(raw_days.strip())
            except ValueError as error:
                msg = f"Invalid horizon day count: {item!r}"
                raise typer.BadParameter(msg) from error
        else:
            label = item.strip()
            if label not in defaults:
                available = ", ".join(defaults)
                msg = f"Unknown horizon {label!r}; use one of {available} or label=days."
                raise typer.BadParameter(msg)
            days = defaults[label]
        if not label or days < 0:
            msg = f"Invalid horizon: {item!r}"
            raise typer.BadParameter(msg)
        horizons.append((label, days))
    return tuple(horizons)


def _strategy_returns_from_results(
    results: dict[str, Any],
    strategy_names: list[str],
) -> dict[str, pd.Series]:
    strategy_returns: dict[str, pd.Series] = {}
    for strategy_name in strategy_names:
        result = results.get(strategy_name)
        returns = getattr(result, "returns", None)
        if returns is None:
            continue
        series = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
        if series.empty:
            continue
        strategy_returns[strategy_name] = series.astype(float)
    return strategy_returns


def _load_scenario_history(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        msg = f"Scenario history file does not exist: {path}"
        raise typer.BadParameter(msg)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if frame.empty:
        return None
    if not _has_scenario_history_date_column(frame):
        console.print(
            "Scenario history has no date column; ignoring it to avoid historical lookahead."
        )
        return None
    return clean_scenario_history(frame)


def _has_scenario_history_date_column(frame: pd.DataFrame) -> bool:
    return any(
        column in frame
        for column in (
            "origin_date",
            "as_of_date",
            "date",
            "snapshot_time",
            "market_date",
            "created_at_utc",
            "created_at",
        )
    )


def _factor_returns_from_snapshot_prices(prices: object) -> pd.DataFrame | None:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        return None
    returns = (
        prices.sort_index()
        .astype(float)
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
    )
    frame = pd.DataFrame(index=returns.index)
    for factor_name, proxy_ticker, _label, _description in DEFAULT_FACTOR_ATTRIBUTION_FACTOR_SPECS:
        if proxy_ticker in returns:
            frame[factor_name] = pd.to_numeric(returns[proxy_ticker], errors="coerce")
    frame = frame.dropna(how="all")
    return frame if not frame.empty else None


def _simulation_ablation_validation(
    returns: pd.Series,
    *,
    scenario_history: pd.DataFrame | None,
    factor_returns: pd.DataFrame | None,
    config: ForwardSimulationValidationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants: list[tuple[str, str, ForwardSimulationValidationConfig, pd.DataFrame | None]] = [
        (
            "baseline_regime_blocks",
            "Baseline regime blocks",
            replace(config, duration_aware_transitions=False, covariate_match_weight=0.0),
            None,
        ),
        (
            "duration_aware",
            "Duration-aware transitions",
            replace(config, duration_aware_transitions=True, covariate_match_weight=0.0),
            None,
        ),
        (
            "duration_covariate",
            "Duration + covariate matching",
            replace(config, duration_aware_transitions=True),
            None,
        ),
    ]
    if factor_returns is not None and not factor_returns.empty:
        variants.append(
            (
                "factor_proxy",
                "Factor-proxy paths",
                replace(config, duration_aware_transitions=True),
                factor_returns,
            )
        )

    for variant, label, variant_config, variant_factors in variants:
        validation = rolling_origin_simulation_backtest(
            returns,
            scenario_history=scenario_history,
            factor_returns=variant_factors,
            config=variant_config,
        )
        summary = summarize_simulation_validation(validation)
        rows.append(
            {
                "variant": variant,
                "label": label,
                "uses_duration_aware_transitions": variant_config.duration_aware_transitions,
                "uses_covariate_matching": variant_config.covariate_match_weight > 0.0,
                "uses_factor_proxy": variant_factors is not None,
                "rows": summary.get("rows"),
                "origins": summary.get("origins"),
                "horizons": summary.get("horizons"),
                "interval_coverage": summary.get("interval_coverage"),
                "target_coverage": summary.get("target_coverage"),
                "coverage_error": summary.get("coverage_error"),
                "median_error_mean": summary.get("median_error_mean"),
                "median_abs_error": summary.get("median_abs_error"),
                "severe_drawdown_brier": summary.get("severe_drawdown_brier"),
                "launch_decision_accuracy": summary.get("launch_decision_accuracy"),
                "launch_action_score": summary.get("launch_action_score"),
                "launch_overrisk_rate": summary.get("launch_overrisk_rate"),
                "constructive_capture_rate": summary.get("constructive_capture_rate"),
                "validity_read": summary.get("validity_read"),
            }
        )
    return pd.DataFrame(rows)


def _slug_identifier(value: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in value.lower())
    return "_".join(part for part in slug.split("_") if part) or "strategy"


def _print_simulation_validation_summary(strategy: str, summary: dict[str, object]) -> None:
    table = Table(title=f"Simulation Validation: {strategy}")
    table.add_column("metric")
    table.add_column("value")
    rows = [
        ("rows", summary.get("rows")),
        ("origins", summary.get("origins")),
        ("horizons", summary.get("horizons")),
        ("interval coverage", _format_optional_percent(summary.get("interval_coverage"))),
        ("target coverage", _format_optional_percent(summary.get("target_coverage"))),
        ("coverage error", _format_optional_percent(summary.get("coverage_error"))),
        ("mean p50 error", _format_optional_percent(summary.get("median_error_mean"))),
        ("median abs error", _format_optional_percent(summary.get("median_abs_error"))),
        ("severe drawdown brier", _format_optional_decimal(summary.get("severe_drawdown_brier"))),
        (
            "launch decision accuracy",
            _format_optional_percent(summary.get("launch_decision_accuracy")),
        ),
        ("launch action score", _format_optional_percent(summary.get("launch_action_score"))),
        ("over-risk rate", _format_optional_percent(summary.get("launch_overrisk_rate"))),
        (
            "constructive capture",
            _format_optional_percent(summary.get("constructive_capture_rate")),
        ),
        ("validity read", summary.get("validity_read")),
    ]
    for metric, value in rows:
        table.add_row(metric, str(value))
    console.print(table)


def _print_simulation_horizon_summary(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    table = Table(title="Simulation Validation By Horizon")
    for column in [
        "horizon",
        "rows",
        "coverage error",
        "median abs error",
        "launch accuracy",
        "action score",
        "over-risk",
        "capture",
        "validity read",
    ]:
        table.add_column(column)
    for _, row in frame.iterrows():
        table.add_row(
            str(row.get("horizon", "")),
            str(row.get("rows", "")),
            _format_optional_percent(row.get("coverage_error")),
            _format_optional_percent(row.get("median_abs_error")),
            _format_optional_percent(row.get("launch_decision_accuracy")),
            _format_optional_percent(row.get("launch_action_score")),
            _format_optional_percent(row.get("launch_overrisk_rate")),
            _format_optional_percent(row.get("constructive_capture_rate")),
            str(row.get("validity_read", "")),
        )
    console.print(table)


def _print_simulation_ablation_summary(frame: pd.DataFrame) -> None:
    table = Table(title="Simulation Model Ablation")
    for column in [
        "variant",
        "rows",
        "coverage error",
        "median abs error",
        "severe brier",
        "launch accuracy",
        "action score",
        "over-risk",
        "validity read",
    ]:
        table.add_column(column)
    for _, row in frame.iterrows():
        table.add_row(
            str(row.get("variant", "")),
            str(row.get("rows", "")),
            _format_optional_percent(row.get("coverage_error")),
            _format_optional_percent(row.get("median_abs_error")),
            _format_optional_decimal(row.get("severe_drawdown_brier")),
            _format_optional_percent(row.get("launch_decision_accuracy")),
            _format_optional_percent(row.get("launch_action_score")),
            _format_optional_percent(row.get("launch_overrisk_rate")),
            str(row.get("validity_read", "")),
        )
    console.print(table)


def _print_strategy_rank_validation_summary(summary: dict[str, object]) -> None:
    table = Table(title="Strategy Rank Validation")
    table.add_column("metric")
    table.add_column("value")
    rows = [
        ("rows", summary.get("rows")),
        ("origin horizons", summary.get("origin_horizons")),
        ("top strategy hit rate", _format_optional_percent(summary.get("top_strategy_hit_rate"))),
        ("mean rank correlation", _format_optional_decimal(summary.get("mean_rank_correlation"))),
        ("mean abs rank error", _format_optional_decimal(summary.get("mean_abs_rank_error"))),
        ("ranking read", summary.get("ranking_read")),
    ]
    for metric, value in rows:
        table.add_row(metric, str(value))
    console.print(table)


def _stop_dashboard_from_pid_file(
    pid_path: Path,
    *,
    port: int | None = None,
    timeout_seconds: float,
    force: bool,
) -> bool:
    stopped = False
    pid = _read_pid_file(pid_path)
    if pid is None or not _process_exists(pid):
        pid_path.unlink(missing_ok=True)
    else:
        stopped = _stop_dashboard_pid(pid, timeout_seconds=timeout_seconds, force=force)
        pid_path.unlink(missing_ok=True)

    if port is not None:
        for port_pid in _listening_pids_on_port(port):
            if port_pid == pid and stopped:
                continue
            if _stop_dashboard_pid(port_pid, timeout_seconds=timeout_seconds, force=force):
                console.print(f"Dashboard port {port} listener PID {port_pid} stopped.")
                stopped = True
    return stopped


def _stop_dashboard_pid(pid: int, *, timeout_seconds: float, force: bool) -> bool:
    if not _process_exists(pid):
        return False
    _signal_process(pid, signal.SIGTERM)
    if not _wait_for_process_exit(pid, timeout_seconds=timeout_seconds):
        if not force:
            console.print(f"Dashboard PID {pid} did not stop within {timeout_seconds:.1f}s.")
            return True
        _signal_process(pid, signal.SIGKILL)
        _wait_for_process_exit(pid, timeout_seconds=2.0)
        console.print(f"Dashboard PID {pid} required SIGKILL.")
    else:
        console.print(f"Dashboard PID {pid} stopped.")
    return True


def _read_pid_file(pid_path: Path) -> int | None:
    try:
        text = pid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        pid = int(text)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _listening_pids_on_port(port: int) -> list[int]:
    if port <= 0:
        return []
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for raw_line in result.stdout.splitlines():
        try:
            pid = int(raw_line.strip())
        except ValueError:
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return pids


def _signal_process(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except PermissionError:
        raise
    except OSError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return


def _wait_for_process_exit(pid: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.10)
    return not _process_exists(pid)


def _format_optional_percent(value: object) -> str:
    raw_value: Any = value
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return "n/a"
    if numeric != numeric:
        return "n/a"
    return f"{numeric:.2%}"


def _format_optional_decimal(value: object) -> str:
    raw_value: Any = value
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return "n/a"
    if numeric != numeric:
        return "n/a"
    return f"{numeric:.2f}"


def _print_migration_table(title: str, results: list[WarehouseMigrationResult]) -> None:
    table = Table(title=title)
    for column in ["artifact", "rows", "table_name"]:
        table.add_column(column)
    for result in results:
        table.add_row(
            str(result.artifact),
            f"{int(result.rows):,}",
            str(result.table_name),
        )
    console.print(table)


def _print_snapshot_backfill_plan(
    plan: SnapshotBackfillPlan,
    *,
    selected_dates: tuple[pd.Timestamp, ...],
    purge_existing: bool,
    plan_only: bool,
) -> None:
    selected_index = pd.DatetimeIndex(selected_dates)
    daily_cutoff = pd.Timestamp(plan.daily_cutoff_date)
    daily_count = int((selected_index >= daily_cutoff).sum()) if len(selected_index) else 0
    weekly_count = len(selected_dates) - daily_count
    table = Table(title="Snapshot Backfill Plan")
    table.add_column("field")
    table.add_column("value")
    table.add_row("mode", "plan only" if plan_only else "run")
    table.add_row("purge existing", str(purge_existing))
    table.add_row("start date", plan.start_date)
    table.add_row("end date", plan.end_date)
    table.add_row("daily cutoff date", plan.daily_cutoff_date)
    table.add_row("weekly frequency", plan.weekly_frequency)
    table.add_row("selected snapshots", f"{len(selected_dates):,}")
    table.add_row("older weekly snapshots", f"{weekly_count:,}")
    table.add_row("recent daily snapshots", f"{daily_count:,}")
    if selected_dates:
        table.add_row("first selected", str(selected_dates[0].date()))
        table.add_row("last selected", str(selected_dates[-1].date()))
    console.print(table)


def _print_prebreak_snapshot_plan(
    plan: pd.DataFrame,
    *,
    lookback_days: int,
    postbreak_days: int,
    weekly_frequency: str,
    skipped_existing: int,
    plan_only: bool,
) -> None:
    table = Table(title="Bubble-Break Snapshot Plan")
    table.add_column("field")
    table.add_column("value")
    table.add_row("mode", "plan only" if plan_only else "run")
    table.add_row("lookback days", f"{lookback_days:,}")
    table.add_row("postbreak days", f"{postbreak_days:,}")
    table.add_row("weekly frequency", weekly_frequency)
    table.add_row("existing market dates checked", f"{skipped_existing:,}")
    table.add_row("selected snapshots", f"{len(plan):,}")
    if not plan.empty:
        table.add_row("first selected", str(plan["market_date"].iloc[0]))
        table.add_row("last selected", str(plan["market_date"].iloc[-1]))
        table.add_row("events", f"{plan['event_name'].nunique():,}")
    console.print(table)
    if not plan.empty:
        preview = Table(title="Bubble-Break Snapshot Preview")
        for column in [
            "event_name",
            "market_date",
            "break_date",
            "days_to_break",
            "postbreak_snapshot",
        ]:
            preview.add_column(column)
        for _, row in plan.head(20).iterrows():
            preview.add_row(
                str(row["event_name"]),
                str(row["market_date"]),
                str(row["break_date"]),
                str(row["days_to_break"]),
            )
        console.print(preview)


def _snapshot_delete_summary(title: str, candidates: pd.DataFrame) -> str:
    if candidates.empty:
        return f"{title}: no snapshots deleted."
    total_mb = float(candidates["artifact_size_bytes"].sum()) / (1024 * 1024)
    return f"{title}: deleted {len(candidates):,} snapshot(s), {total_mb:,.1f} MB."


def _selected_daily_market_day_count(
    selected_dates: tuple[pd.Timestamp, ...],
    *,
    daily_cutoff_date: str,
) -> int:
    if not selected_dates:
        return 0
    selected_index = pd.DatetimeIndex(selected_dates).normalize()
    daily_cutoff = pd.Timestamp(daily_cutoff_date).normalize()
    return int((selected_index >= daily_cutoff).sum())


def _print_snapshot_manifest(manifest: SnapshotManifest) -> None:
    table = Table(title="Saved Baseline Snapshot")
    for column in [
        "run_id",
        "created_at_utc",
        "market_date",
        "risk_status",
        "recommended_action",
        "risk_budget_multiplier",
        "artifact_path",
    ]:
        table.add_column(column)
    table.add_row(
        manifest.run_id,
        manifest.created_at_utc,
        manifest.market_date,
        manifest.risk_status,
        manifest.recommended_action,
        f"{manifest.risk_budget_multiplier:.2f}",
        manifest.artifact_path,
    )
    console.print(table)


def _print_daily_update_summary(
    *,
    manifest: SnapshotManifest,
    report_path: Path,
    experiment_dir: Path,
    registry_rows: int,
    migration_results: list[WarehouseMigrationResult],
    valuation_rows: int,
    refresh_data: bool,
    refresh_macro: bool,
    refresh_news: bool,
) -> None:
    migrated_rows = sum(result.rows for result in migration_results)
    summary = Table(title="Daily Update Stack")
    summary.add_column("step")
    summary.add_column("status")
    summary.add_column("detail")
    summary.add_row(
        "Market/macro/news refresh",
        "complete",
        (
            f"prices={'refreshed' if refresh_data else 'cached'}; "
            f"macro={'refreshed' if refresh_macro else 'cached'}; "
            f"news={'refreshed' if refresh_news else 'cached'}"
        ),
    )
    summary.add_row(
        "Snapshot",
        "complete",
        f"{manifest.run_id} | market date {manifest.market_date} | {manifest.risk_status}",
    )
    summary.add_row("Report", "complete", str(report_path))
    summary.add_row(
        "Strategy registry",
        "complete",
        f"{registry_rows:,} snapshot strategies refreshed",
    )
    summary.add_row(
        "Warehouse migration",
        "complete",
        f"{migrated_rows:,} rows from {experiment_dir} and journal tables",
    )
    summary.add_row(
        "Paper valuations",
        "complete",
        f"{valuation_rows:,} rows written for active monitoring windows",
    )
    console.print(summary)


def _print_macro_alignment_summary(summary: dict[str, object], output_dir: Path) -> None:
    table = Table(title="42 Macro / Trade-Bot Alignment")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("status", str(summary.get("status", "unknown")))
    table.add_row("comparisons", f"{int(summary.get('comparisons', 0)):,}")
    if summary.get("status") == "ok":
        table.add_row("date range", f"{summary.get('date_min')} to {summary.get('date_max')}")
        table.add_row(
            "mean 42 posture",
            _format_optional_decimal(summary.get("macro_mean_posture_score")),
        )
        table.add_row(
            "mean trade-bot posture",
            _format_optional_decimal(summary.get("trade_bot_mean_posture_score")),
        )
        table.add_row(
            "mean abs disagreement",
            _format_optional_decimal(summary.get("mean_abs_disagreement")),
        )
        table.add_row("major mismatches", f"{int(summary.get('major_mismatches', 0)):,}")
        table.add_row(
            "large-change major mismatches",
            f"{int(summary.get('large_change_major_mismatches', 0)):,}",
        )
    console.print(table)
    console.print(f"Reports: {output_dir}")


def _print_macro_outcome_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    table = Table(title="42 Macro / Trade-Bot Forward Outcome Scores")
    table.add_column("scope")
    table.add_column("horizon")
    table.add_column("rows", justify="right")
    table.add_column("42 action", justify="right")
    table.add_column("bot action", justify="right")
    table.add_column("42 proxy ret", justify="right")
    table.add_column("bot proxy ret", justify="right")
    if not summary.empty:
        display = summary[
            (summary["scope"].isin(["transcript", "all"]))
            & (summary["horizon"].isin(["1w", "1m", "3m"]))
        ].copy()
        for _, row in display.iterrows():
            table.add_row(
                str(row["scope"]),
                str(row["horizon"]),
                f"{int(row['rows']):,}",
                _format_optional_decimal(row.get("macro_mean_action_score")),
                _format_optional_decimal(row.get("trade_bot_mean_action_score")),
                _format_optional_percent(row.get("macro_mean_proxy_return")),
                _format_optional_percent(row.get("trade_bot_mean_proxy_return")),
            )
    console.print(table)
    console.print(f"Outcome reports: {output_dir}")


def _active_experiment_dir(experiment_dir: Path) -> Path:
    if experiment_dir == DEFAULT_EXPERIMENTS_DIR and DEFAULT_RESET_EXPERIMENTS_DIR.exists():
        return DEFAULT_RESET_EXPERIMENTS_DIR
    return experiment_dir


if __name__ == "__main__":
    app()
