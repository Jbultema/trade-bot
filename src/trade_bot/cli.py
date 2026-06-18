from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from trade_bot.config import configured_tickers, load_config
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.DEFAULT import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_MACRO_PATH,
    DEFAULT_NEWS_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
)
from trade_bot.reporting.report import write_baseline_report
from trade_bot.research.baselines import run_configured_baselines
from trade_bot.research.experiments import run_experiment_iteration
from trade_bot.storage.run_store import RunStore, SnapshotManifest

app = typer.Typer(no_args_is_help=True)
console = Console()


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
            f"{row['promotion_score']:.2f}",
            f"{row['cagr']:.2%}",
            f"{row['sharpe']:.2f}",
            f"{row['max_drawdown']:.2%}",
            f"{row['calmar']:.2f}",
            f"{row['worst_3y_cagr']:.2%}",
        )
    console.print(table)
    console.print(f"Wrote experiment outputs to {Path(output_dir) / f'iteration_{iteration:02d}'}")


def _format_optional_percent(value: object) -> str:
    raw_value: Any = value
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return "n/a"
    if numeric != numeric:
        return "n/a"
    return f"{numeric:.2%}"


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


if __name__ == "__main__":
    app()
