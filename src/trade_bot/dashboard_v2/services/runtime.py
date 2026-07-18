from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st

from trade_bot.config import load_config
from trade_bot.dashboard.loaders import (
    load_live_run,
    load_snapshot_dashboard_run,
    load_snapshot_dashboard_run_by_id,
    load_snapshot_jobs_frame,
)
from trade_bot.DEFAULTS import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_FORWARD_TEST_STRATEGY,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_MACRO_PATH,
    DEFAULT_NEWS_PATH,
    DEFAULT_RUN_STORE_ARTIFACT_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_RUN_STORE_JOB_LOG_DIR,
)
from trade_bot.research.action_headline import ActionHeadline, build_action_headline
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.strategy_decision import resolve_trade_decision_for_strategy
from trade_bot.research.strategy_naming import strategy_display_name
from trade_bot.research.trade_decision import TradeDecisionRun
from trade_bot.storage.run_store import RunStore, SnapshotManifest
from trade_bot.trading.book_alignment import (
    BookAlignmentRun,
    build_book_alignment,
)
from trade_bot.trading.journal import JournalBook, TradeJournal

RunSource = Literal["Latest snapshot (fast)", "Selected snapshot", "Live pipeline"]


@dataclass(frozen=True)
class DashboardPaths:
    config_path: Path = DEFAULT_CONFIG_PATH
    events_path: Path = DEFAULT_EVENTS_PATH
    macro_path: Path = DEFAULT_MACRO_PATH
    news_path: Path = DEFAULT_NEWS_PATH
    journal_path: Path = DEFAULT_JOURNAL_PATH
    run_store_path: Path = DEFAULT_RUN_STORE_DB_PATH
    artifact_dir: Path = DEFAULT_RUN_STORE_ARTIFACT_DIR
    job_log_dir: Path = DEFAULT_RUN_STORE_JOB_LOG_DIR


@dataclass(frozen=True)
class DashboardRuntime:
    paths: DashboardPaths
    run_source: RunSource
    bot_config: object
    baseline_run: BaselineRun
    snapshot_manifest: SnapshotManifest | None
    snapshot_loaded: bool
    journal: TradeJournal
    selected_book: JournalBook
    promoted_book: JournalBook
    operating_trade_decision: TradeDecisionRun
    open_ticket_count: int
    book_alignment: BookAlignmentRun
    execution_book_alignment: BookAlignmentRun | None
    action_headline: ActionHeadline


@dataclass(frozen=True)
class BookSelection:
    selected_book: JournalBook
    promoted_book: JournalBook


def render_path_controls(defaults: DashboardPaths | None = None) -> DashboardPaths:
    defaults = defaults or DashboardPaths()
    with st.sidebar.expander("Local paths", expanded=False):
        return DashboardPaths(
            config_path=Path(st.text_input("Config", str(defaults.config_path))),
            events_path=Path(st.text_input("Events", str(defaults.events_path))),
            macro_path=Path(st.text_input("Macro", str(defaults.macro_path))),
            news_path=Path(st.text_input("News", str(defaults.news_path))),
            journal_path=Path(st.text_input("Trade journal", str(defaults.journal_path))),
            run_store_path=Path(st.text_input("Run store", str(defaults.run_store_path))),
            artifact_dir=Path(st.text_input("Snapshot artifacts", str(defaults.artifact_dir))),
            job_log_dir=Path(st.text_input("Snapshot job logs", str(defaults.job_log_dir))),
        )


def render_book_selector(
    journal_path: Path,
    *,
    baseline_run: BaselineRun | None = None,
    bot_config: object | None = None,
) -> BookSelection:
    journal = TradeJournal(journal_path)
    books = journal.list_books()
    promoted_book = journal.get_promoted_book()
    options = books["book_id"].astype(str).tolist()
    strategy_options = _strategy_selector_options(
        current_values=books.get("strategy_name", pd.Series(dtype=str)).astype(str).tolist(),
        baseline_run=baseline_run,
        bot_config=bot_config,
    )
    session_key = "dashboard_v2_selected_book_id"
    selected_book_id = str(st.session_state.get(session_key, promoted_book.book_id))
    if selected_book_id not in options:
        selected_book_id = promoted_book.book_id

    st.markdown("### Operating book")
    cols = st.columns([2, 1, 1])

    def _book_label(book_id: str) -> str:
        row = books[books["book_id"].astype(str).eq(str(book_id))].iloc[0]
        prefix = "Promoted: " if int(row.get("is_promoted", 0)) == 1 else ""
        return f"{prefix}{row['book_name']} ({row['mode']}/{row['account']})"

    selected_book_id = cols[0].selectbox(
        "Book selector",
        options,
        index=options.index(selected_book_id),
        format_func=_book_label,
        key=session_key,
        help=(
            "Select the named book to inspect in Forward Test. Today uses the promoted "
            "book for main-line alerts and headline recommendations."
        ),
    )
    selected_book = journal.get_book(str(selected_book_id))
    cols[1].metric("Promoted Book", promoted_book.book_name)
    if cols[2].button(
        "Set Selected as Promoted",
        disabled=selected_book.book_id == promoted_book.book_id,
        help="Make the selected book the default operating book for Today.",
    ):
        journal.set_promoted_book(selected_book.book_id)
        st.rerun()

    with st.expander("Create named book", expanded=False):
        form_defaults = selected_book
        with st.form("dashboard_v2_create_book_form"):
            create_cols = st.columns(5)
            book_name = create_cols[0].text_input("Book name", f"{form_defaults.book_name} Copy")
            mode = create_cols[1].selectbox(
                "Mode",
                ["paper", "live"],
                index=0 if form_defaults.mode == "paper" else 1,
            )
            account = create_cols[2].text_input("Account", f"{form_defaults.account}_copy")
            strategy_name = _strategy_selectbox(
                create_cols[3],
                "Strategy to follow",
                form_defaults.strategy_name,
                strategy_options,
                key="dashboard_v2_create_book_strategy",
            )
            account_value = create_cols[4].number_input(
                "Account value",
                min_value=1.0,
                value=float(form_defaults.account_value),
                step=1000.0,
            )
            st.caption(
                "Strategy to follow controls the target decision used for this book when "
                "that strategy is available in the loaded run; existing journal-only labels "
                "remain selectable for record continuity."
            )
            promote = st.checkbox("Promote this new book", value=False)
            if st.form_submit_button("Create Book"):
                new_book_id = journal.upsert_book(
                    book_name=book_name,
                    mode=mode,
                    account=account,
                    strategy_name=strategy_name,
                    account_value=float(account_value),
                    promote=bool(promote),
                )
                st.session_state[session_key] = new_book_id
                st.rerun()

    with st.expander("Book controls", expanded=False):
        st.caption(
            "Edit, promote, or delete named book configurations. Deleting removes only the "
            "book selector entry; existing ticket and execution records stay in the journal."
        )
        st.dataframe(
            books[
                [
                    "book_name",
                    "mode",
                    "account",
                    "strategy_name",
                    "account_value",
                    "is_promoted",
                    "updated_at_utc",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        managed_book_id = st.selectbox(
            "Book to manage",
            options,
            index=options.index(selected_book.book_id),
            format_func=_book_label,
            key="dashboard_v2_managed_book_id",
        )
        managed_book = journal.get_book(str(managed_book_id))
        with st.form("dashboard_v2_manage_book_form"):
            manage_cols = st.columns(5)
            managed_name = manage_cols[0].text_input("Book name", managed_book.book_name)
            managed_mode = manage_cols[1].selectbox(
                "Mode",
                ["paper", "live"],
                index=0 if managed_book.mode == "paper" else 1,
            )
            managed_account = manage_cols[2].text_input("Account", managed_book.account)
            managed_strategy = _strategy_selectbox(
                manage_cols[3],
                "Strategy to follow",
                managed_book.strategy_name,
                strategy_options,
                key=f"dashboard_v2_manage_book_strategy_{managed_book.book_id}",
            )
            managed_account_value = manage_cols[4].number_input(
                "Account value",
                min_value=1.0,
                value=float(managed_book.account_value),
                step=1000.0,
            )
            control_cols = st.columns([1, 1, 3])
            save_changes = control_cols[0].form_submit_button("Save Changes")
            promote_managed = control_cols[1].form_submit_button(
                "Promote",
                disabled=managed_book.is_promoted,
            )
            if save_changes:
                journal.upsert_book(
                    book_id=managed_book.book_id,
                    book_name=managed_name,
                    mode=managed_mode,
                    account=managed_account,
                    strategy_name=managed_strategy,
                    account_value=float(managed_account_value),
                    promote=managed_book.is_promoted,
                )
                st.session_state[session_key] = managed_book.book_id
                st.rerun()
            if promote_managed:
                journal.upsert_book(
                    book_id=managed_book.book_id,
                    book_name=managed_name,
                    mode=managed_mode,
                    account=managed_account,
                    strategy_name=managed_strategy,
                    account_value=float(managed_account_value),
                    promote=True,
                )
                st.session_state[session_key] = managed_book.book_id
                st.rerun()
        delete_confirm = st.checkbox(
            f"Confirm delete '{managed_book.book_name}'",
            value=False,
            disabled=managed_book.is_promoted,
            key=f"dashboard_v2_delete_book_confirm_{managed_book.book_id}",
        )
        if st.button(
            "Delete Book",
            disabled=managed_book.is_promoted or not delete_confirm,
            help=(
                "Promoted books cannot be deleted. Promote another book first. "
                "Deletion removes only the named-book configuration."
            ),
        ):
            journal.delete_book(managed_book.book_id)
            st.session_state[session_key] = promoted_book.book_id
            st.rerun()
    return BookSelection(selected_book=selected_book, promoted_book=promoted_book)


def _strategy_selectbox(
    container: object,
    label: str,
    current_value: str,
    option_frame: pd.DataFrame,
    *,
    key: str,
) -> str:
    values = _strategy_option_values(option_frame, current_value)
    selected_value = str(current_value or "").strip() or values[0]
    if selected_value not in values:
        values.insert(0, selected_value)
    return str(
        container.selectbox(
            label,
            values,
            index=values.index(selected_value),
            format_func=lambda value: _strategy_option_label(option_frame, str(value)),
            key=key,
            help=(
                "Select the strategy target this book should follow. The default "
                "scenario-adjusted decision uses the configured primary strategy."
            ),
        )
    )


def _strategy_option_values(option_frame: pd.DataFrame, current_value: str) -> list[str]:
    if option_frame.empty or "strategy_name" not in option_frame:
        values: list[str] = []
    else:
        values = option_frame["strategy_name"].astype(str).tolist()
    current = str(current_value or "").strip()
    if current and current not in values:
        values.append(current)
    if not values:
        values.append(DEFAULT_FORWARD_TEST_STRATEGY)
    return values


def _strategy_option_label(option_frame: pd.DataFrame, strategy_name: str) -> str:
    if not option_frame.empty and "strategy_name" in option_frame:
        matches = option_frame[option_frame["strategy_name"].astype(str).eq(str(strategy_name))]
        if not matches.empty:
            return str(matches.iloc[0].get("label", strategy_name))
    return f"Existing journal value | {strategy_name}"


def _strategy_selector_options(
    *,
    current_values: list[str],
    baseline_run: BaselineRun | None = None,
    bot_config: object | None = None,
) -> pd.DataFrame:
    rows: dict[str, dict[str, object]] = {}

    def add(strategy_name: str, label: str, priority: int) -> None:
        strategy_name = str(strategy_name or "").strip()
        if not strategy_name:
            return
        existing = rows.get(strategy_name)
        if existing is None or int(existing["priority"]) > priority:
            rows[strategy_name] = {
                "strategy_name": strategy_name,
                "label": label,
                "priority": priority,
            }

    primary_strategy = str(getattr(bot_config, "primary_strategy", "") or "").strip()
    default_label = "Scenario-adjusted trade decision"
    if primary_strategy:
        default_label = f"{default_label} | primary: {strategy_display_name(primary_strategy)}"
    add(DEFAULT_FORWARD_TEST_STRATEGY, default_label, 0)
    if primary_strategy:
        add(
            primary_strategy,
            f"Primary configured strategy | {strategy_display_name(primary_strategy)}",
            1,
        )

    runtime_results = getattr(baseline_run, "results", {}) if baseline_run is not None else {}
    for strategy_name in runtime_results:
        strategy_name = str(strategy_name)
        add(strategy_name, f"Configured strategy | {strategy_display_name(strategy_name)}", 10)

    for strategy_name in current_values:
        add(str(strategy_name), f"Existing journal value | {strategy_name}", 20)

    if not rows:
        add(DEFAULT_FORWARD_TEST_STRATEGY, "Scenario-adjusted trade decision", 0)

    return (
        pd.DataFrame(rows.values())
        .sort_values(["priority", "strategy_name"])
        .reset_index(drop=True)
    )


def snapshot_choices(paths: DashboardPaths, *, limit: int = 100) -> pd.DataFrame:
    store = RunStore(paths.run_store_path, artifact_dir=paths.artifact_dir, job_log_dir=paths.job_log_dir)
    return store.list_snapshots(limit=limit)


def snapshot_option_label(frame: pd.DataFrame, run_id: str) -> str:
    row = frame[frame["run_id"].astype(str) == str(run_id)].iloc[0]
    return f"{row['market_date']} | {row['risk_status']} | {row['created_at_utc']} | {str(run_id)[:18]}"


def load_runtime(
    *,
    paths: DashboardPaths,
    run_source: RunSource,
    selected_snapshot_run_id: str | None,
    selected_book: JournalBook | None = None,
    promoted_book: JournalBook | None = None,
    refresh_data: bool = False,
    refresh_macro: bool = False,
    refresh_news: bool = False,
) -> DashboardRuntime:
    bot_config = load_config(paths.config_path)
    snapshot_manifest: SnapshotManifest | None = None
    snapshot_loaded = False

    if run_source == "Latest snapshot (fast)":
        snapshot_payload = load_snapshot_dashboard_run(
            str(paths.config_path),
            str(paths.events_path),
            str(paths.macro_path),
            str(paths.news_path),
            str(paths.run_store_path),
            str(paths.artifact_dir),
            str(paths.job_log_dir),
        )
        if snapshot_payload is None:
            baseline_run = load_live_run(
                str(paths.config_path),
                str(paths.events_path),
                str(paths.macro_path),
                str(paths.news_path),
                refresh_data,
                refresh_macro,
                refresh_news,
            )
        else:
            baseline_run, snapshot_manifest = snapshot_payload
            snapshot_loaded = True
    elif run_source == "Selected snapshot" and selected_snapshot_run_id:
        baseline_run, snapshot_manifest = load_snapshot_dashboard_run_by_id(
            str(paths.run_store_path),
            str(paths.artifact_dir),
            str(paths.job_log_dir),
            selected_snapshot_run_id,
        )
        snapshot_loaded = True
    else:
        baseline_run = load_live_run(
            str(paths.config_path),
            str(paths.events_path),
            str(paths.macro_path),
            str(paths.news_path),
            refresh_data,
            refresh_macro,
            refresh_news,
        )

    journal = TradeJournal(paths.journal_path)
    promoted_book = promoted_book or journal.get_promoted_book()
    selected_book = selected_book or promoted_book
    operating_trade_decision = resolve_trade_decision_for_strategy(
        baseline_run,
        promoted_book.strategy_name,
    )
    open_tickets = journal.load_recommendation_tickets(
        status="open",
        mode=promoted_book.mode,
        account=promoted_book.account,
        strategy_name=promoted_book.strategy_name,
    )
    book_alignment = build_book_alignment(
        journal=journal,
        trade_decision=operating_trade_decision,
        prices=baseline_run.prices,
        mode=promoted_book.mode,
        account=promoted_book.account,
        strategy_name=promoted_book.strategy_name,
        account_value=promoted_book.account_value,
    )
    execution_book_alignment = (
        book_alignment if not book_alignment.position_plan.empty else None
    )
    action_headline = build_action_headline(
        current_state=baseline_run.current_state,
        trade_decision=operating_trade_decision,
        news_monitor=baseline_run.news_monitor,
        open_ticket_count=len(open_tickets),
        position_plan=book_alignment.position_plan,
    )
    return DashboardRuntime(
        paths=paths,
        run_source=run_source,
        bot_config=bot_config,
        baseline_run=baseline_run,
        snapshot_manifest=snapshot_manifest,
        snapshot_loaded=snapshot_loaded,
        journal=journal,
        selected_book=selected_book,
        promoted_book=promoted_book,
        operating_trade_decision=operating_trade_decision,
        open_ticket_count=len(open_tickets),
        book_alignment=book_alignment,
        execution_book_alignment=execution_book_alignment,
        action_headline=action_headline,
    )


def freshness_label(runtime: DashboardRuntime) -> str:
    manifest = runtime.snapshot_manifest
    if manifest is None:
        return f"{runtime.run_source} live at {datetime.now(UTC).replace(microsecond=0).isoformat()}"
    return f"{manifest.market_date} | {manifest.risk_status.upper()} | {manifest.created_at_utc}"


def load_job_frame(paths: DashboardPaths) -> pd.DataFrame:
    return load_snapshot_jobs_frame(
        str(paths.run_store_path),
        str(paths.artifact_dir),
        str(paths.job_log_dir),
    )
