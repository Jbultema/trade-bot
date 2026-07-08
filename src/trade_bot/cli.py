from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from trade_bot.config import configured_tickers, load_config
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.DEFAULTS import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_FACTOR_ATTRIBUTION_FACTOR_SPECS,
    DEFAULT_FORWARD_SIMULATION_BLOCK_DAYS,
    DEFAULT_FORWARD_SIMULATION_PATHS,
    DEFAULT_FORWARD_SIMULATION_VALIDATION_HORIZONS,
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
)
from trade_bot.ml.diagnostics import run_ml_diagnostics
from trade_bot.reporting.report import write_baseline_report
from trade_bot.research.baselines import run_configured_baselines
from trade_bot.research.entry_date_analysis import build_entry_date_analysis
from trade_bot.research.experiment_monitor import (
    load_experiment_candidates,
    load_experiment_scorecards,
)
from trade_bot.research.experiments import run_experiment_iteration
from trade_bot.research.forward_simulation import (
    ForwardSimulationValidationConfig,
    rolling_origin_simulation_backtest,
    rolling_origin_strategy_rank_validation,
    summarize_simulation_validation,
    summarize_strategy_rank_validation,
)
from trade_bot.research.signal_evidence import (
    build_signal_family_evidence,
    build_signal_family_marginal_tests,
    tag_scorecard_signal_families,
)
from trade_bot.storage.run_store import RunStore, SnapshotManifest
from trade_bot.storage.warehouse import TradingWarehouse, WarehouseMigrationResult

app = typer.Typer(no_args_is_help=True)
console = Console()
DEFAULT_SIMULATION_VALIDATION_REFERENCE_STRATEGIES = ",".join(
    name for name, _label in DEFAULT_SIMULATION_REFERENCE_STRATEGIES
)


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
    keep_latest: Annotated[int, typer.Option("--keep-latest")] = 12,
    keep_per_market_date: Annotated[int, typer.Option("--keep-per-market-date")] = 2,
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
        apply=apply,
    )
    mode = "Applied" if apply else "Dry run"
    if candidates.empty:
        console.print(
            f"{mode}: no snapshots fall outside the retention policy "
            f"(keep_latest={keep_latest}, keep_per_market_date={keep_per_market_date})."
        )
        return

    total_bytes = int(candidates["artifact_size_bytes"].sum())
    total_mb = total_bytes / (1024 * 1024)
    console.print(
        f"{mode}: {len(candidates):,} snapshot(s), {total_mb:,.1f} MB outside "
        f"retention policy (keep_latest={keep_latest}, "
        f"keep_per_market_date={keep_per_market_date})."
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
    scenario_history: Annotated[
        Path | None,
        typer.Option(
            "--scenario-history",
            help="Optional date-stamped CSV or parquet scenario probabilities.",
        ),
    ] = None,
    ablation: Annotated[
        bool,
        typer.Option(
            "--ablation/--skip-ablation",
            help="Write a model-ablation readout comparing baseline, duration, covariate, and factor-proxy variants.",
        ),
    ] = False,
) -> None:
    """Validate forward simulation calibration against historical realized paths."""

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
    if scenario_history_frame is None:
        console.print(
            "No date-stamped scenario history supplied; validation uses the empirical "
            "return-regime library and fallback scenario probabilities."
        )
    else:
        console.print(
            f"Loaded {len(scenario_history_frame):,} scenario-history rows from "
            f"{scenario_history}."
        )

    config = ForwardSimulationValidationConfig(
        origin_frequency=origin_frequency,
        horizons=_parse_validation_horizons(horizons),
        min_train_days=min_train_days,
        paths=paths,
        block_days=block_days,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = rolling_origin_simulation_backtest(
        returns_by_strategy[strategy],
        scenario_history=scenario_history_frame,
        config=config,
    )
    validation_summary = summarize_simulation_validation(validation)
    validation_path = output_dir / f"{_slug_identifier(strategy)}_simulation_validation.csv"
    validation.to_csv(validation_path, index=False)

    console.print(
        f"Validated {strategy} from snapshot {manifest.run_id} "
        f"({manifest.market_date}); wrote {validation_path}."
    )
    _print_simulation_validation_summary(strategy, validation_summary)

    ablation_frame: pd.DataFrame | None = None
    ablation_path_string = ""
    if ablation:
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
        scenario_history_path=str(scenario_history or ""),
        validation_output_path=str(validation_path),
        ablation_output_path=ablation_path_string,
        rank_output_path=rank_path_string,
        validation_summary=validation_summary,
        validation=validation,
        ablation_summary=ablation_frame,
    )
    console.print(f"Saved simulation validation history to DuckDB as {validation_run_id}.")


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
    return frame


def _has_scenario_history_date_column(frame: pd.DataFrame) -> bool:
    return any(
        column in frame
        for column in ("origin_date", "as_of_date", "date", "created_at_utc", "created_at")
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
        ("validity read", summary.get("validity_read")),
    ]
    for metric, value in rows:
        table.add_row(metric, str(value))
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


def _active_experiment_dir(experiment_dir: Path) -> Path:
    if experiment_dir == DEFAULT_EXPERIMENTS_DIR and DEFAULT_RESET_EXPERIMENTS_DIR.exists():
        return DEFAULT_RESET_EXPERIMENTS_DIR
    return experiment_dir


if __name__ == "__main__":
    app()
