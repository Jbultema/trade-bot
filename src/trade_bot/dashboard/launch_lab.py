from __future__ import annotations

import html
from datetime import date
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.components import (
    _clearable_selectbox,
    _helped_metric,
    _render_metric_dataframe,
    _render_runtime_notice,
)
from trade_bot.dashboard.decision_sniff import (
    build_operational_sniff_read,
    render_operational_sniff_read,
)
from trade_bot.dashboard.formatting import (
    _display_metrics,
    _format_currency,
    _format_decimal,
    _format_percent,
)
from trade_bot.DEFAULTS import (
    DEFAULT_ENTRY_HORIZONS,
    DEFAULT_LAUNCH_AGGREGATE_MAX_STRATEGIES,
    DEFAULT_LAUNCH_CAPITAL,
    DEFAULT_LAUNCH_PRIMARY_HORIZON,
    DEFAULT_LAUNCH_PROTOCOL_MATERIAL_SPREAD,
    DEFAULT_LAUNCH_PROTOCOL_SMALL_SPREAD,
    DEFAULT_LAUNCH_START_FREQUENCY,
    DEFAULT_LAUNCH_TARGET_FRACTION,
)
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.experiment_operator import (
    build_experiment_operator_plan,
)
from trade_bot.research.launch_readiness import (
    AggregateLaunchReadinessRun,
    LaunchReadinessRun,
    build_aggregate_launch_readiness,
    build_launch_readiness,
)
from trade_bot.storage.warehouse import TradingWarehouse


def _render_launch_lab(
    bot_config: Any,
    baseline_run: BaselineRun,
    experiment_scorecards: pd.DataFrame,
    *,
    warehouse_path: str = "",
) -> None:
    st.subheader("Launch Lab")
    st.caption(
        "Entry-gate workbench for deciding whether a paper strategy should move to a small "
        "pseudo-live or live sleeve now. This is separate from daily book alignment for "
        "strategies that are already running."
    )

    from trade_bot.dashboard.simulation_lab import _result_for_strategy, _strategy_option_frame

    options = _strategy_option_frame(
        bot_config=bot_config,
        baseline_run=baseline_run,
        experiment_scorecards=experiment_scorecards,
    )
    selected_strategy, selected_result = _selected_launch_strategy(
        bot_config=bot_config,
        baseline_run=baseline_run,
        options=options,
        result_loader=_result_for_strategy,
    )
    if selected_strategy is None or selected_result is None:
        st.info("Choose a strategy to evaluate launch timing.")
        return

    input_cols = st.columns([1.0, 0.75, 0.9, 0.85, 0.9])
    benchmark_options = _benchmark_options(baseline_run)
    benchmark_name = input_cols[0].selectbox(
        "Launch benchmark",
        benchmark_options,
        index=0,
        help=(
            "Reference for start-date beat rates. For the high-growth i111 family, QQQ is the "
            "default hurdle; BIL/cash remains the floor and SPY is secondary context."
        ),
        key="launch_lab_benchmark",
    )
    horizon_options = list(DEFAULT_ENTRY_HORIZONS)
    primary_horizon = input_cols[1].selectbox(
        "Entry horizon",
        horizon_options,
        index=_default_option_index(horizon_options, DEFAULT_LAUNCH_PRIMARY_HORIZON),
        help=(
            "Forward window used for launch evidence. Shorter horizons show entry timing; "
            "longer horizons show whether adoption timing mattered after initial noise."
        ),
        key="launch_lab_primary_horizon",
    )
    start_frequency = _start_frequency_selector(input_cols[2])
    capital_to_launch = input_cols[3].number_input(
        "Test capital",
        min_value=0.0,
        value=float(DEFAULT_LAUNCH_CAPITAL),
        step=100.0,
        help="Dollar sleeve you are considering for a new paper/live launch.",
        key="launch_lab_capital",
    )
    target_fraction = input_cols[4].slider(
        "Final sleeve fraction",
        min_value=0.10,
        max_value=1.00,
        value=float(DEFAULT_LAUNCH_TARGET_FRACTION),
        step=0.05,
        help="Fraction of the test capital that should be deployed when the launch ramp is complete.",
        key="launch_lab_target_fraction",
    )

    benchmark_result = baseline_run.results.get(benchmark_name)
    run = build_launch_readiness(
        selected_result,
        benchmark_result=benchmark_result,
        current_state=baseline_run.current_state,
        capital_to_launch=capital_to_launch,
        target_fraction=target_fraction,
        primary_horizon=str(primary_horizon),
        start_frequency=start_frequency,
    )
    _render_launch_decision(run, selected_strategy, benchmark_name)
    render_operational_sniff_read(
        build_operational_sniff_read(
            baseline_run=baseline_run,
            bot_config=bot_config,
            strategy_name=selected_strategy,
            result=selected_result,
            benchmark_ticker=benchmark_name,
        ),
        title="Selected Strategy Sniff Test",
        include_details=True,
        expanded_details=False,
    )

    launch_view = (
        st.pills(
            "Launch Lab view",
            [
                "Experiment Operator",
                "Aggregate View",
                "Why / Why Not",
                "Entry Backtest",
                "Ramp Plan",
                "How to Use This",
            ],
            selection_mode="single",
            default="Experiment Operator",
            key="launch_lab_view",
            width="stretch",
        )
        or "Experiment Operator"
    )
    _render_launch_view_runtime_notice(launch_view)

    if launch_view == "Experiment Operator":
        _render_experiment_operator(
            selected_strategy=selected_strategy,
            selected_result=selected_result,
            run=run,
            warehouse_path=warehouse_path,
        )
    elif launch_view == "Aggregate View":
        with st.spinner("Building aggregate launch read across candidate strategies..."):
            aggregate_run = _build_aggregate_launch_run(
                bot_config=bot_config,
                baseline_run=baseline_run,
                options=options,
                result_loader=_result_for_strategy,
                benchmark_result=benchmark_result,
                current_state=baseline_run.current_state,
                start_frequency=start_frequency,
                target_fraction=target_fraction,
            )
        _render_aggregate_launch_lab(aggregate_run, str(primary_horizon))
    elif launch_view == "Why / Why Not":
        _render_launch_gate(run)
    elif launch_view == "Entry Backtest":
        _render_entry_backtest(run)
    elif launch_view == "Ramp Plan":
        _render_ramp_plan(run)
    else:
        _render_launch_vs_operating()


def _selected_launch_strategy(
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    options: pd.DataFrame,
    result_loader: Any,
) -> tuple[str | None, BacktestResult | None]:
    if options.empty:
        return None, None
    label_column = "simulation_label" if "simulation_label" in options else "strategy"
    labels = options[label_column].astype(str).tolist()
    selected_label = _clearable_selectbox(
        "Strategy to launch-check",
        labels,
        key="launch_lab_selected_strategy",
        placeholder="Search launch candidates...",
    )
    if selected_label is None:
        return None, None
    row = options[options[label_column].astype(str).eq(str(selected_label))].iloc[0]
    strategy_name = str(row["strategy"])
    return strategy_name, result_loader(
        strategy_name,
        bot_config=bot_config,
        baseline_run=baseline_run,
    )


def _build_aggregate_launch_run(
    *,
    bot_config: Any,
    baseline_run: BaselineRun,
    options: pd.DataFrame,
    result_loader: Any,
    benchmark_result: BacktestResult | None,
    current_state: Any,
    start_frequency: str,
    target_fraction: float,
) -> AggregateLaunchReadinessRun:
    aggregate_options = _aggregate_launch_option_frame(options)
    strategy_results: dict[str, BacktestResult] = {}
    for strategy_name in aggregate_options.get("strategy", pd.Series(dtype=str)).astype(str):
        result = result_loader(
            strategy_name,
            bot_config=bot_config,
            baseline_run=baseline_run,
        )
        if result is not None:
            strategy_results[strategy_name] = result
    return build_aggregate_launch_readiness(
        strategy_results,
        benchmark_result=benchmark_result,
        current_state=current_state,
        start_frequency=start_frequency,
        target_fraction=target_fraction,
    )


def _render_launch_view_runtime_notice(launch_view: str) -> None:
    if launch_view == "Aggregate View":
        _render_runtime_notice(
            "Aggregate View can be slow",
            (
                "This view rebuilds launch-readiness summaries across the curated candidate "
                "set and multiple horizons. Changing horizon, frequency, or sleeve settings "
                "will rerun that aggregate pass."
            ),
            tone="warning",
        )
    elif launch_view == "Entry Backtest":
        _render_runtime_notice(
            "Entry Backtest renders the densest launch table",
            (
                "This view is usually faster than aggregate launch, but changing the strategy "
                "or start frequency can rebuild many historical start-window rows."
            ),
            tone="neutral",
        )


def _aggregate_launch_option_frame(options: pd.DataFrame) -> pd.DataFrame:
    columns = ["strategy"]
    if options.empty or "strategy" not in options:
        return pd.DataFrame(columns=columns)
    frame = options.copy()
    eligibility_masks = []
    for column in ["is_growth_pareto_efficient", "is_pareto_efficient", "pareto_frontier"]:
        if column in frame:
            eligibility_masks.append(frame[column].fillna(False).astype(bool))
    for column in ["curation_rank", "curated_rank"]:
        if column in frame:
            eligibility_masks.append(pd.to_numeric(frame[column], errors="coerce").notna())
    if "growth_utility_tier" in frame:
        eligibility_masks.append(
            frame["growth_utility_tier"]
            .astype(str)
            .str.contains("champion|challenger", case=False, na=False)
        )
    if eligibility_masks:
        eligible = frame[pd.concat(eligibility_masks, axis=1).any(axis=1)].copy()
        if not eligible.empty:
            frame = eligible

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, sort_ascending in [
        ("curation_rank", True),
        ("curated_rank", True),
        ("growth_constrained_utility_score", False),
        ("promotion_score", False),
        ("cagr", False),
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            sort_columns.append(column)
            ascending.append(sort_ascending)
    if sort_columns:
        frame = frame.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
        )
    return frame.drop_duplicates("strategy").head(DEFAULT_LAUNCH_AGGREGATE_MAX_STRATEGIES)


def _render_aggregate_launch_lab(
    aggregate_run: AggregateLaunchReadinessRun,
    primary_horizon: str,
) -> None:
    st.markdown("**Aggregate Launch Read**")
    st.caption(
        "Cross-strategy entry-gate evidence across curated and Pareto candidates. Use this to see "
        "whether launch guidance is strategy-specific or broadly shifts from wait to set/ready as "
        "the horizon extends."
    )
    if aggregate_run.strategy_count <= 0:
        st.info("No curated or Pareto strategies were available for aggregate launch analysis.")
        return

    _render_aggregate_launch_cards(aggregate_run, primary_horizon)
    chart_cols = st.columns([1.0, 1.15])
    with chart_cols[0]:
        st.plotly_chart(
            _horizon_label_count_figure(aggregate_run.horizon_label_counts),
            use_container_width=True,
        )
    with chart_cols[1]:
        st.plotly_chart(
            _horizon_transition_figure(aggregate_run.horizon_transition_matrix),
            use_container_width=True,
        )
    st.plotly_chart(
        _protocol_separation_figure(aggregate_run.protocol_separation_by_horizon),
        use_container_width=True,
    )

    with st.expander("Aggregate launch detail tables", expanded=False):
        detail_view = (
            st.pills(
                "Aggregate detail table",
                [
                    "Best Horizon Rows",
                    "Transition Matrix",
                    "Protocol Separation",
                    "Protocol by Horizon",
                ],
                selection_mode="single",
                default="Best Horizon Rows",
                key="launch_lab_aggregate_detail_view",
                width="stretch",
            )
            or "Best Horizon Rows"
        )
        if detail_view == "Best Horizon Rows":
            _render_metric_dataframe(
                _display_metrics(aggregate_run.strategy_horizon_summary),
                hide_index=True,
            )
        elif detail_view == "Transition Matrix":
            _render_metric_dataframe(
                _display_metrics(aggregate_run.horizon_transition_matrix),
                hide_index=True,
            )
        elif detail_view == "Protocol Separation":
            _render_metric_dataframe(
                _display_metrics(aggregate_run.protocol_separation),
                hide_index=True,
            )
        else:
            _render_metric_dataframe(
                _display_metrics(aggregate_run.protocol_separation_by_horizon),
                hide_index=True,
            )


def _render_aggregate_launch_cards(
    aggregate_run: AggregateLaunchReadinessRun,
    primary_horizon: str,
) -> None:
    counts = aggregate_run.horizon_label_counts
    selected_counts = (
        counts[counts["horizon"].astype(str).eq(primary_horizon)].copy()
        if not counts.empty and "horizon" in counts
        else pd.DataFrame()
    )
    set_ready = _launch_label_count(selected_counts, {"set", "ready"})
    wait_no_go = _launch_label_count(selected_counts, {"wait", "no_go"})
    transitions = aggregate_run.horizon_transition_matrix
    upgrades = _transition_count(transitions, "upgrade")
    downgrades = _transition_count(transitions, "downgrade")
    protocol_by_horizon = aggregate_run.protocol_separation_by_horizon
    selected_protocol = (
        protocol_by_horizon[protocol_by_horizon["horizon"].astype(str).eq(primary_horizon)]
        if not protocol_by_horizon.empty and "horizon" in protocol_by_horizon
        else pd.DataFrame()
    )
    protocol_row = (
        selected_protocol.iloc[0].to_dict() if not selected_protocol.empty else {}
    )
    material_rate = _safe_float(protocol_row.get("material_separation_rate"))
    median_spread = _safe_float(protocol_row.get("median_protocol_spread"))
    cards = [
        {
            "label": "Strategies Evaluated",
            "answer": str(aggregate_run.strategy_count),
            "detail": (
                "Curated, Pareto, or high-utility candidates included in the aggregate launch shelf."
            ),
            "tone": "neutral",
        },
        {
            "label": f"{primary_horizon} Set / Ready",
            "answer": f"{set_ready}/{aggregate_run.strategy_count}",
            "detail": (
                f"{wait_no_go} candidate(s) remain wait/no-go at the selected horizon."
            ),
            "tone": "success" if set_ready > wait_no_go else "warning",
        },
        {
            "label": "Horizon Upgrades",
            "answer": f"{upgrades} up / {downgrades} down",
            "detail": (
                "Counts how often candidate launch labels improve or deteriorate as the evidence "
                "window extends from one horizon to the next."
            ),
            "tone": "success" if upgrades >= downgrades else "warning",
        },
        {
            "label": "Ramp Separation",
            "answer": _format_percent(material_rate),
            "detail": (
                f"{_format_percent(median_spread)} median protocol spread at {primary_horizon}; "
                "material means 4w/8w/12w ramps changed median outcomes enough to matter."
            ),
            "tone": "success" if material_rate >= 0.25 else "warning",
        },
    ]
    _render_launch_card_grid(cards, class_name="launch-guidance-grid")


def _benchmark_options(baseline_run: BaselineRun) -> list[str]:
    preferred = ["buy_hold_qqq", "buy_hold_bil", "buy_hold_spy", "i41_ref_us_60_40"]
    options = [name for name in preferred if name in baseline_run.results]
    return options or sorted(baseline_run.results.keys())[:1]


def _default_option_index(options: list[str], default: str) -> int:
    try:
        return options.index(default)
    except ValueError:
        return 0


def _start_frequency_selector(container: Any) -> str:
    options = {
        "Monthly starts": "M",
        "Quarterly starts": "Q",
        "Annual starts": "A",
    }
    default_label = next(
        (
            label
            for label, value in options.items()
            if value == DEFAULT_LAUNCH_START_FREQUENCY
        ),
        "Monthly starts",
    )
    selected = container.selectbox(
        "Start sampling",
        list(options),
        index=list(options).index(default_label),
        help=(
            "How historical launch dates are sampled. Monthly starts give more observations "
            "but overlap heavily for 3m+ horizons; annual starts are a lower-overlap sanity check."
        ),
        key="launch_lab_start_frequency",
    )
    return options[str(selected)]


def _render_launch_decision(
    run: LaunchReadinessRun,
    selected_strategy: str,
    benchmark_name: str,
) -> None:
    recommendation = run.recommendation
    launch_label = str(recommendation.get("launch_label", "wait")).replace("_", " ").title()
    launch_action = str(recommendation.get("launch_action", "Review launch evidence."))
    launch_read = str(recommendation.get("launch_read", "No launch read available."))
    tone = _launch_tone(str(recommendation.get("launch_label", "wait")))
    st.markdown(
        f"""
        <div class="action-callout action-callout-{tone}">
            <p class="action-kicker">Launch Readiness</p>
            <h3>{html.escape(launch_label)}: {html.escape(launch_action)}</h3>
            <p>{html.escape(launch_read)}</p>
            <p><strong>Boundary:</strong> Launch Lab is for new or scale-up capital in
            <code>{html.escape(selected_strategy)}</code>. Once the sleeve is running, use Book Alignment,
            tickets, and Forward Test for daily target drift.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_launch_snapshot_cards(run, benchmark_name)
    _render_launch_evidence_cards(run)


def _render_launch_gate(run: LaunchReadinessRun) -> None:
    st.markdown("**Current Launch Gate**")
    st.caption(
        "Current-state diagnostics for fresh capital. This is the answer to: "
        "is today a good entry setup, or should the strategy stay on deck?"
    )
    _render_current_gate_read(run)
    if run.diagnostics.empty:
        st.info("No current-state diagnostics are available.")
    else:
        with st.expander("Current launch diagnostics", expanded=False):
            _render_metric_dataframe(_display_metrics(run.diagnostics), hide_index=True)

    if not run.summary.empty:
        primary_horizon = str(run.recommendation.get("horizon", DEFAULT_LAUNCH_PRIMARY_HORIZON))
        frame = run.summary[run.summary["horizon"].astype(str).eq(primary_horizon)].copy()
        with st.expander("Protocol comparison for the primary horizon", expanded=False):
            _render_metric_dataframe(_display_metrics(frame), hide_index=True)


def _render_entry_backtest(run: LaunchReadinessRun) -> None:
    st.markdown("**Entry Backtest**")
    st.caption(
        "Historical start-date evidence. Each row asks what happened when the strategy was "
        "adopted from a sampled historical start date using a specific ramp schedule."
    )
    if run.windows.empty:
        st.info("No launch windows are available for the selected strategy.")
        return
    primary_horizon = str(run.recommendation.get("horizon", DEFAULT_LAUNCH_PRIMARY_HORIZON))
    windows = run.windows[run.windows["horizon"].astype(str).eq(primary_horizon)].copy()
    if windows.empty:
        windows = run.windows.copy()

    _render_entry_backtest_read(run, windows)
    chart_cols = st.columns([1.15, 1.0])
    with chart_cols[0]:
        st.plotly_chart(_entry_scatter(windows), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(_protocol_bar(run.summary, primary_horizon), use_container_width=True)

    with st.expander("Launch start-date detail", expanded=False):
        columns = [
            "protocol",
            "start_date",
            "end_date",
            "total_return",
            "benchmark_return",
            "excess_return",
            "max_drawdown",
            "first_month_drawdown",
            "bad_start",
        ]
        _render_metric_dataframe(
            _display_metrics(windows[[column for column in columns if column in windows]]),
            hide_index=True,
        )


def _render_ramp_plan(run: LaunchReadinessRun) -> None:
    st.markdown("**Suggested Ramp Plan**")
    st.caption(
        "Dollar schedule for the intended test sleeve. Re-check Launch Lab before adding each staged tranche."
    )
    if run.ramp_plan.empty:
        st.info("No ramp plan is available.")
        return
    _render_ramp_plan_read(run)
    cols = st.columns(4)
    latest = run.ramp_plan.iloc[-1]
    _helped_metric(cols[0], "Final Deployment", _format_currency(latest.get("capital_deployed")))
    _helped_metric(cols[1], "Final Sleeve", _format_percent(latest.get("account_fraction_deployed")))
    _helped_metric(cols[2], "Ramp Weeks", str(int(latest.get("week", 0) or 0)))
    _helped_metric(cols[3], "Reserved Cash", _format_currency(latest.get("cash_reserved")))
    st.plotly_chart(_ramp_figure(run.ramp_plan), use_container_width=True)
    _render_metric_dataframe(_display_metrics(run.ramp_plan), hide_index=True)


def _render_experiment_operator(
    *,
    selected_strategy: str,
    selected_result: BacktestResult,
    run: LaunchReadinessRun,
    warehouse_path: str,
) -> None:
    st.markdown("**Experiment Operator**")
    st.caption(
        "Paper/live trial contract for answering: if fresh cash starts following this strategy now, "
        "how long must it run and what evidence validates or fails the launch?"
    )
    controls = st.columns([0.45, 0.9, 0.65])
    mode = controls[0].radio(
        "Experiment mode",
        ["paper", "live"],
        horizontal=True,
        key="launch_experiment_mode",
    )
    account = controls[1].text_input(
        "Account label",
        "default_paper_account" if str(mode) == "paper" else "default_live_account",
        key="launch_experiment_account",
    )
    start_date = controls[2].date_input(
        "Experiment start",
        date.today(),
        key="launch_experiment_start_date",
        help="This is the monitoring-window start date used to judge the experiment.",
    )

    windows, valuations = _load_launch_monitoring_frames(warehouse_path)
    plan = build_experiment_operator_plan(
        selected_result,
        launch_run=run,
        mode=str(mode),
        account=str(account),
        monitoring_windows=windows,
        valuations=valuations,
    )
    _render_experiment_operator_cards(plan)
    st.info(plan.status_read)

    st.markdown("**Success Contract**")
    _render_metric_dataframe(plan.success_contract, hide_index=True)
    st.markdown("**Checkpoint Contract**")
    _render_metric_dataframe(plan.checkpoint_contract, hide_index=True)
    if not plan.current_status.empty:
        with st.expander("Current matching monitoring read", expanded=False):
            _render_metric_dataframe(_display_metrics(plan.current_status), hide_index=True)

    action_cols = st.columns([0.75, 1.0])
    with action_cols[0]:
        st.markdown("**Start or Update Experiment Monitoring**")
        st.caption(
            "This creates or updates a normal Monitoring window, so the experiment flows into "
            "the same champion/challenger valuation tables used elsewhere."
        )
        if st.button("Start / Update Experiment Window", type="primary"):
            if not warehouse_path:
                st.error("No warehouse path is configured for monitoring.")
            else:
                try:
                    result = TradingWarehouse(warehouse_path).monitor_strategy(
                        selected_strategy,
                        role="challenger",
                        mode=plan.mode,
                        account=plan.account,
                        capital_base=plan.recommended_capital,
                        start_date=start_date.isoformat(),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    _load_launch_monitoring_frames.clear()
                    st.success(
                        f"Monitoring {result.strategy_name} as an experiment challenger "
                        f"with {_format_currency(plan.recommended_capital)}."
                    )
                    st.rerun()
    with action_cols[1]:
        st.markdown("**Operator Notes**")
        st.write(
            f"Use **{plan.required_horizon}** as the first decision horizon. "
            f"The primary hurdle is **{plan.primary_benchmark}** with **{plan.cash_floor}** "
            f"as the cash floor; **{plan.secondary_benchmark}** is context only."
        )
        st.write(plan.capital_rationale)


def _render_experiment_operator_cards(plan: Any) -> None:
    cards = []
    for _, row in plan.summary_cards.iterrows():
        metric = str(row.get("metric", ""))
        value = row.get("value")
        if metric == "Trial capital":
            answer = _format_currency(value)
            tone = "success" if float(value or 0.0) >= 5_000 else "warning"
        elif metric == "Launch confidence":
            answer = _format_decimal(value)
            tone = "success" if float(value or 0.0) >= 0.65 else "warning"
        elif metric == "Signal cycle":
            answer = f"{int(value or 0)} trading days"
            tone = "neutral"
        elif metric == "Current status":
            answer = str(value).replace("_", " ").title()
            tone = _experiment_status_tone(str(value))
        else:
            answer = str(value).replace("_", " ").title()
            tone = "neutral"
        cards.append(
            {
                "label": metric,
                "answer": answer,
                "detail": str(row.get("detail", "")),
                "tone": tone,
            }
        )
    _render_launch_card_grid(cards, class_name="launch-summary-grid")


@st.cache_data(ttl=60, show_spinner=False)
def _load_launch_monitoring_frames(warehouse_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not warehouse_path:
        return pd.DataFrame(), pd.DataFrame()
    warehouse = TradingWarehouse(warehouse_path)
    return warehouse.list_monitoring_windows(status=None), warehouse.read_table(
        "strategy_daily_valuations"
    )


def _experiment_status_tone(status: str) -> str:
    if status == "validate":
        return "success"
    if status == "fail":
        return "critical"
    if status == "continue":
        return "warning"
    return "neutral"


def _render_launch_vs_operating() -> None:
    st.markdown("**Launching vs Operating**")
    st.caption(
        "Use this when Launch Lab appears to conflict with the daily recommendation. "
        "It usually means one area is judging new capital while the other is managing a running book."
    )
    _render_launch_card_grid(
        [
            {
                "label": "New capital",
                "answer": "Launch Lab",
                "detail": "Answers whether a new paper/live sleeve should start now, wait, or stage in gradually.",
                "tone": "warning",
            },
            {
                "label": "Running book",
                "answer": "Book Alignment and Forward Test",
                "detail": "Answers what an already-running paper/live book should hold after the daily target update.",
                "tone": "success",
            },
            {
                "label": "Scale-up capital",
                "answer": "Treat it as a new tranche",
                "detail": "Run Launch Lab for the incremental dollars, then log any executed tranche in Forward Test.",
                "tone": "neutral",
            },
        ],
        class_name="launch-guidance-grid",
    )
    _render_metric_dataframe(
        pd.DataFrame(
            [
                {
                    "mode": "Launch behavior",
                    "question": "Should I start or scale this strategy now?",
                    "uses": "Entry backtests, current risk posture, and staged-ramp evidence.",
                    "output": "Ready / Set / Wait / No-go plus a ramp plan.",
                    "what_not_to_do": "Do not treat launch staging as the target weight for an already-running book.",
                },
                {
                    "mode": "Operating behavior",
                    "question": "What should the running paper/live book hold today?",
                    "uses": "Daily target weights, book alignment, tickets, and execution logs.",
                    "output": "Do nothing, small rebalance, or material trade tickets.",
                    "what_not_to_do": "Do not wait for a new launch signal every time the strategy rebalances.",
                },
                {
                    "mode": "Scale-up behavior",
                    "question": "Should I add more capital to a strategy I already trust?",
                    "uses": "Launch Lab for the new tranche and Monitoring for live/paper drift.",
                    "output": "Add a tranche now, stage in, or keep collecting evidence.",
                    "what_not_to_do": "Do not confuse adding new capital with correcting target drift on existing capital.",
                },
            ]
        ),
        hide_index=True,
    )


def _render_launch_snapshot_cards(run: LaunchReadinessRun, benchmark_name: str) -> None:
    recommendation = run.recommendation
    cards = [
        {
            "label": "Launch Score",
            "answer": _format_decimal(recommendation.get("launch_score")),
            "detail": "Blend of historical entry outcomes and current launch-gate conditions.",
            "tone": "neutral",
        },
        {
            "label": "Recommended Protocol",
            "answer": str(recommendation.get("protocol", "n/a")),
            "detail": "The best protocol for the selected horizon after entry-history and current-state scoring.",
            "tone": "warning",
        },
        {
            "label": "Horizon",
            "answer": str(recommendation.get("horizon", DEFAULT_LAUNCH_PRIMARY_HORIZON)),
            "detail": "The forward window used for the headline entry-readiness calculation.",
            "tone": "neutral",
        },
        {
            "label": "Bad-Start Rate",
            "answer": _format_percent(recommendation.get("bad_start_rate")),
            "detail": "Share of historical launch windows that lost money or had an early drawdown breach.",
            "tone": "critical" if _safe_float(recommendation.get("bad_start_rate"), 0.0) >= 0.25 else "neutral",
        },
        {
            "label": "Beat Rate",
            "answer": _format_percent(recommendation.get("beat_rate")),
            "detail": "Share of sampled starts where the launch protocol beat the selected benchmark.",
            "tone": "success" if _safe_float(recommendation.get("beat_rate"), 0.0) >= 0.60 else "warning",
        },
        {
            "label": "Benchmark",
            "answer": benchmark_name,
            "detail": "The hurdle used to judge whether the launch window was worth taking.",
            "tone": "neutral",
        },
    ]
    _render_launch_card_grid(cards, class_name="launch-summary-grid")


def _render_launch_evidence_cards(run: LaunchReadinessRun) -> None:
    evidence = _launch_evidence(run)
    launch_label = str(run.recommendation.get("launch_label", "wait"))
    blocker_tone = "success" if launch_label == "ready" else "critical"
    _render_launch_card_grid(
        [
            {
                "label": "Current blockers",
                "answer": evidence["against_answer"],
                "detail": evidence["against_detail"],
                "tone": blocker_tone,
            },
            {
                "label": "Why keep watching",
                "answer": evidence["support_answer"],
                "detail": evidence["support_detail"],
                "tone": "success",
            },
            {
                "label": "What to do instead",
                "answer": evidence["next_answer"],
                "detail": evidence["next_detail"],
                "tone": "warning",
            },
        ],
        class_name="launch-guidance-grid",
    )


def _render_current_gate_read(run: LaunchReadinessRun) -> None:
    diagnostics = run.diagnostics
    if diagnostics.empty:
        return
    friction = _diagnostics_with_impact(diagnostics, {"launch_friction", "stage_in_bias"})
    support = _diagnostics_with_impact(diagnostics, {"supports_launch"})
    context = _diagnostics_with_impact(diagnostics, {"context", "neutral"})
    cards = [
        {
            "label": "Entry Friction",
            "answer": _gate_answer(friction, "No major current-state friction"),
            "detail": _diagnostic_detail(friction, limit=3)
            or "Current risk and strategy conditions are not forcing a worse launch read.",
            "tone": "critical" if not friction.empty else "success",
        },
        {
            "label": "Launch Support",
            "answer": _gate_answer(support, "No strong launch support"),
            "detail": _diagnostic_detail(support, limit=3)
            or "The current setup has not produced enough positive confirmation to upgrade the launch gate.",
            "tone": "success" if not support.empty else "warning",
        },
        {
            "label": "Next Investigation",
            "answer": _what_changes_the_launch_answer(run),
            "detail": "Use this as the short checklist before committing fresh capital; operating books still follow Book Alignment.",
            "tone": "warning",
        },
        {
            "label": "Context",
            "answer": _gate_answer(context, "No additional context"),
            "detail": _diagnostic_detail(context, limit=2)
            or "No extra context diagnostics are available for this strategy and current state.",
            "tone": "neutral",
        },
    ]
    _render_launch_card_grid(cards, class_name="launch-guidance-grid")


def _render_entry_backtest_read(run: LaunchReadinessRun, windows: pd.DataFrame) -> None:
    recommendation = run.recommendation
    if windows.empty:
        return
    protocol = str(recommendation.get("protocol", "selected protocol"))
    protocol_windows = windows[windows["protocol"].astype(str).eq(protocol)].copy()
    if protocol_windows.empty:
        protocol_windows = windows.copy()
    best_start = protocol_windows.sort_values("total_return", ascending=False).iloc[0]
    worst_start = protocol_windows.sort_values("total_return", ascending=True).iloc[0]
    protocol_read = _protocol_separation_read(windows)
    overlap_read = _window_overlap_read(windows)
    cards = [
        {
            "label": "Historical Pattern",
            "answer": (
                f"{_format_percent(recommendation.get('positive_return_rate'))} positive, "
                f"{_format_percent(recommendation.get('bad_start_rate'))} bad starts"
            ),
            "detail": (
                "Dots below zero are starts that lost money over the selected window. "
                "Larger dots are starts with an early drawdown breach."
            ),
            "tone": "warning",
        },
        {
            "label": "Window Overlap",
            "answer": overlap_read["answer"],
            "detail": overlap_read["detail"],
            "tone": overlap_read["tone"],
        },
        {
            "label": "Protocol Separation",
            "answer": protocol_read["answer"],
            "detail": protocol_read["detail"],
            "tone": protocol_read["tone"],
        },
        {
            "label": "Best Start",
            "answer": f"{best_start.get('start_date')} -> {_format_percent(best_start.get('total_return'))}",
            "detail": f"Beat benchmark by {_format_percent(best_start.get('excess_return'))}.",
            "tone": "success",
        },
        {
            "label": "Worst Start",
            "answer": f"{worst_start.get('start_date')} -> {_format_percent(worst_start.get('total_return'))}",
            "detail": (
                f"Max drawdown was {_format_percent(worst_start.get('max_drawdown'))}; "
                f"excess return was {_format_percent(worst_start.get('excess_return'))}."
            ),
            "tone": "critical",
        },
    ]
    _render_launch_card_grid(cards, class_name="launch-guidance-grid")


def _protocol_separation_read(windows: pd.DataFrame) -> dict[str, str]:
    if windows.empty or "protocol" not in windows or "total_return" not in windows:
        return {
            "answer": "No protocol spread",
            "detail": "Protocol separation cannot be calculated for this window.",
            "tone": "neutral",
        }
    medians = (
        windows.assign(total_return=pd.to_numeric(windows["total_return"], errors="coerce"))
        .dropna(subset=["total_return"])
        .groupby("protocol")["total_return"]
        .median()
        .sort_values(ascending=False)
    )
    if medians.empty or len(medians) < 2:
        return {
            "answer": "Single protocol",
            "detail": "Only one launch protocol is available for this selected horizon.",
            "tone": "neutral",
        }
    spread = float(medians.max() - medians.min())
    best = str(medians.index[0])
    worst = str(medians.index[-1])
    if spread < DEFAULT_LAUNCH_PROTOCOL_SMALL_SPREAD:
        answer = "Protocols effectively identical"
        tone = "warning"
        detail = (
            f"Median-return spread is only {_format_percent(spread)} across protocols. "
            "For this horizon, the ramp period is too small relative to the full window to change the read much."
        )
    elif spread < DEFAULT_LAUNCH_PROTOCOL_MATERIAL_SPREAD:
        answer = f"{_format_percent(spread)} median-return spread"
        tone = "neutral"
        detail = (
            f"Best median protocol is {best}; weakest is {worst}. "
            "The ramp choice matters a little, but the strategy path still dominates."
        )
    else:
        answer = f"{_format_percent(spread)} median-return spread"
        tone = "success"
        detail = (
            f"Best median protocol is {best}; weakest is {worst}. "
            "The launch ramp is materially changing outcomes for this horizon."
        )
    return {"answer": answer, "detail": detail, "tone": tone}


def _window_overlap_read(windows: pd.DataFrame) -> dict[str, str]:
    if windows.empty or "start_date" not in windows:
        return {
            "answer": "Overlap unknown",
            "detail": "Start-date overlap cannot be calculated for this window.",
            "tone": "neutral",
        }
    unique_starts = (
        pd.to_datetime(windows["start_date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    if len(unique_starts) < 2:
        return {
            "answer": "Single start sample",
            "detail": "There are not enough sampled starts to assess overlap.",
            "tone": "neutral",
        }
    horizon_days = None
    if "horizon_trading_days" in windows:
        horizon_series = pd.to_numeric(windows["horizon_trading_days"], errors="coerce").dropna()
        if not horizon_series.empty:
            horizon_days = float(horizon_series.iloc[0])
    if not horizon_days or horizon_days <= 0:
        return {
            "answer": "Overlap unknown",
            "detail": "The selected launch windows do not include horizon length.",
            "tone": "neutral",
        }
    median_gap_calendar = float(unique_starts.diff().dt.days.dropna().median())
    estimated_gap_trading = median_gap_calendar * 5.0 / 7.0
    overlap = max(0.0, min(1.0, 1.0 - estimated_gap_trading / horizon_days))
    if overlap >= 0.60:
        tone = "warning"
        detail = (
            f"Sampled starts are roughly {int(round(median_gap_calendar))} calendar days apart "
            f"against a {int(round(horizon_days))}-trading-day horizon. Adjacent dots share most "
            "of the same future market path, so waves can look seasonal even when they are overlapping-window math."
        )
    elif overlap >= 0.25:
        tone = "neutral"
        detail = (
            f"Sampled starts have moderate overlap: roughly {int(round(median_gap_calendar))} calendar days "
            f"between starts against a {int(round(horizon_days))}-trading-day horizon."
        )
    else:
        tone = "success"
        detail = (
            f"Sampled starts are mostly independent for this horizon: roughly {int(round(median_gap_calendar))} "
            f"calendar days between starts against a {int(round(horizon_days))}-trading-day horizon."
        )
    return {
        "answer": f"{_format_percent(overlap)} estimated overlap",
        "detail": detail,
        "tone": tone,
    }


def _render_ramp_plan_read(run: LaunchReadinessRun) -> None:
    recommendation = run.recommendation
    label = str(recommendation.get("launch_label", "wait"))
    plan = run.ramp_plan
    if plan.empty:
        return
    first = plan.iloc[0]
    last = plan.iloc[-1]
    if label in {"wait", "no_go"}:
        answer = "Keep the intended sleeve in reserve"
        detail = (
            "No deployment is scheduled until the launch gate improves. Keep monitoring the "
            "candidate rather than forcing an entry."
        )
        tone = "critical"
    elif label == "set":
        answer = "Starter sleeve only"
        detail = (
            f"Deploy {_format_currency(first.get('capital_deployed'))} first and keep "
            f"{_format_currency(first.get('cash_reserved'))} reserved until the launch gate confirms."
        )
        tone = "warning"
    else:
        answer = "Launch plan is cleared"
        detail = (
            f"Final deployment reaches {_format_currency(last.get('capital_deployed'))}; "
            "still re-check before executing if the daily risk state changes."
        )
        tone = "success"
    _render_launch_card_grid(
        [
            {
                "label": "Ramp Read",
                "answer": answer,
                "detail": detail,
                "tone": tone,
            },
            {
                "label": "Execution Rule",
                "answer": "Re-check before each tranche",
                "detail": "A staged ramp is an entry protocol for new dollars, not a substitute for daily target drift management.",
                "tone": "neutral",
            },
        ],
        class_name="launch-guidance-grid",
    )


def _launch_evidence(run: LaunchReadinessRun) -> dict[str, str]:
    recommendation = run.recommendation
    label = str(recommendation.get("launch_label", "wait"))
    friction = _diagnostics_with_impact(run.diagnostics, {"launch_friction", "stage_in_bias"})
    support = _diagnostics_with_impact(run.diagnostics, {"supports_launch"})
    bad_start = _safe_float(recommendation.get("bad_start_rate"), 0.0)
    beat_rate = _safe_float(recommendation.get("beat_rate"), 0.0)
    median_excess = _safe_float(recommendation.get("median_excess_return"), 0.0)
    worst_excess = _safe_float(recommendation.get("worst_excess_return"), 0.0)
    against_reasons = []
    if bad_start >= 0.25:
        against_reasons.append(f"bad-start rate is {_format_percent(bad_start)}")
    if beat_rate < 0.60:
        against_reasons.append(f"beat rate is only {_format_percent(beat_rate)}")
    if worst_excess < -0.20:
        against_reasons.append(f"worst excess window is {_format_percent(worst_excess)}")
    if not friction.empty:
        against_reasons.append(_diagnostic_detail(friction, limit=2))
    support_reasons = []
    if beat_rate >= 0.55:
        support_reasons.append(f"beat rate is {_format_percent(beat_rate)}")
    if median_excess > 0:
        support_reasons.append(f"median excess return is {_format_percent(median_excess)}")
    if not support.empty:
        support_reasons.append(_diagnostic_detail(support, limit=2))
    if label == "ready":
        next_answer = "Use the ramp plan"
        next_detail = "The launch gate is supportive enough to open the intended sleeve, subject to manual review."
    elif label == "set":
        next_answer = "Starter sleeve or wait"
        next_detail = (
            "If you want live learning, use only a small starter tranche. Otherwise keep the strategy "
            "on deck until risk-off probability, transition pressure, or bearish confirmations improve."
        )
    else:
        next_answer = "Do not open new capital today"
        next_detail = (
            "Keep the candidate in paper monitoring, rerun after the next daily update, and look for "
            "current risk to improve before launching."
        )
    return {
        "against_answer": " | ".join(part for part in against_reasons if part) or "No single blocker",
        "against_detail": (
            "These are the reasons the entry gate is not giving a clean launch read."
            if against_reasons
            else "No major hard blocker was detected; use the support and ramp cards to decide whether this is ready or only a starter setup."
        ),
        "support_answer": " | ".join(part for part in support_reasons if part) or "Candidate remains worth monitoring",
        "support_detail": "These are reasons to keep the strategy on deck instead of pruning it.",
        "next_answer": next_answer,
        "next_detail": next_detail,
    }


def _diagnostics_with_impact(diagnostics: pd.DataFrame, impacts: set[str]) -> pd.DataFrame:
    if diagnostics.empty or "score_impact" not in diagnostics:
        return pd.DataFrame()
    return diagnostics[diagnostics["score_impact"].astype(str).isin(impacts)].copy()


def _diagnostic_detail(frame: pd.DataFrame, *, limit: int) -> str:
    if frame.empty:
        return ""
    parts = []
    for _, row in frame.head(limit).iterrows():
        parts.append(f"{row.get('read')}: {_format_diagnostic_value(row.get('metric'), row.get('value'))}")
    return "; ".join(parts)


def _gate_answer(frame: pd.DataFrame, fallback: str) -> str:
    if frame.empty:
        return fallback
    return f"{len(frame)} signal(s): " + ", ".join(frame["read"].astype(str).head(2).tolist())


def _what_changes_the_launch_answer(run: LaunchReadinessRun) -> str:
    diagnostics = dict(zip(run.diagnostics["metric"], run.diagnostics["value"], strict=False))
    asks = []
    risk_status = str(diagnostics.get("risk_status", "")).upper()
    risk_off = _safe_float(diagnostics.get("risk_off_1m_probability"), 0.0)
    transition = _safe_float(diagnostics.get("transition_1m_probability"), 0.0)
    one_month = _safe_float(diagnostics.get("strategy_return_1m"), 0.0)
    if risk_status in {"ORANGE", "RED"}:
        asks.append("risk status improves")
    if risk_off >= 0.20:
        asks.append("1M risk-off falls below 20%")
    if transition >= 0.35:
        asks.append("transition pressure cools")
    if one_month <= 0.0:
        asks.append("strategy 1M return turns positive")
    return ", ".join(asks[:3]) if asks else "No obvious launch blocker remains"


def _format_diagnostic_value(metric: object, value: object) -> str:
    metric_name = str(metric)
    if metric_name == "risk_status":
        return str(value).upper()
    if any(token in metric_name for token in ["probability", "share", "return", "drawdown"]):
        return _format_percent(value)
    if metric_name == "risk_score":
        return _format_decimal(value)
    return str(value)


def _horizon_label_count_figure(counts: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not counts.empty:
        frame = counts.copy()
        frame = frame.sort_values(["horizon_order", "launch_label"])
        for label in _launch_label_order():
            label_frame = frame[frame["launch_label"].astype(str).eq(label)]
            fig.add_trace(
                go.Bar(
                    x=label_frame["horizon"],
                    y=label_frame["count"],
                    name=_launch_label_display(label),
                    marker={"color": _launch_label_colors().get(label, "#64748b")},
                    customdata=label_frame[["share"]],
                    hovertemplate=(
                        "%{x}<br>"
                        f"{_launch_label_display(label)}: %{{y}} strategies<br>"
                        "Share %{customdata[0]:.1%}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        barmode="stack",
        height=360,
        title="Launch labels by horizon",
        xaxis_title="Entry horizon",
        yaxis_title="Strategy count",
        margin={"l": 20, "r": 20, "t": 48, "b": 86},
        legend={"orientation": "h", "yanchor": "top", "y": -0.24, "xanchor": "left", "x": 0.0},
    )
    return fig


def _horizon_transition_figure(transitions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if transitions.empty:
        fig.update_layout(
            height=360,
            title="Launch-label transitions by horizon",
            margin={"l": 20, "r": 20, "t": 48, "b": 64},
        )
        return fig

    frame = transitions.copy()
    horizon_pairs = frame["horizon_pair"].drop_duplicates().astype(str).tolist()
    transition_labels = _transition_label_order(frame)
    pivot = (
        frame.pivot_table(
            index="transition",
            columns="horizon_pair",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=transition_labels, columns=horizon_pairs, fill_value=0)
        .astype(float)
    )
    hover = (
        frame.assign(
            hover=frame.apply(
                lambda row: (
                    f"{row['transition']}<br>{row['horizon_pair']}<br>"
                    f"Count {int(row['count'])}<br>"
                    f"Direction {row['direction']}<br>"
                    f"Examples {row['example_strategies'] or 'n/a'}"
                ),
                axis=1,
            )
        )
        .pivot_table(
            index="transition",
            columns="horizon_pair",
            values="hover",
            aggfunc="first",
            fill_value="",
        )
        .reindex(index=transition_labels, columns=horizon_pairs, fill_value="")
    )
    fig.add_trace(
        go.Heatmap(
            x=horizon_pairs,
            y=transition_labels,
            z=pivot.to_numpy(),
            customdata=hover.to_numpy(),
            colorscale="Teal",
            colorbar={"title": "count"},
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        title="Horizon transition matrix",
        xaxis_title="Horizon step",
        yaxis_title="Launch label transition",
        margin={"l": 20, "r": 20, "t": 48, "b": 64},
    )
    return fig


def _protocol_separation_figure(protocol_by_horizon: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not protocol_by_horizon.empty:
        frame = protocol_by_horizon.sort_values("horizon_order").copy()
        rate_specs = [
            ("material_separation_rate", "Material", "#0f766e"),
            ("small_separation_rate", "Small", "#f59e0b"),
            ("effectively_identical_rate", "Effectively identical", "#64748b"),
        ]
        for column, label, color in rate_specs:
            if column not in frame:
                continue
            fig.add_trace(
                go.Bar(
                    x=frame["horizon"],
                    y=frame[column],
                    name=label,
                    marker={"color": color},
                    customdata=frame[["median_protocol_spread", "strategies"]],
                    hovertemplate=(
                        "%{x}<br>"
                        f"{label}: %{{y:.1%}}<br>"
                        "Median spread %{customdata[0]:.1%}<br>"
                        "Strategies %{customdata[1]}<extra></extra>"
                    ),
                )
            )
        if "median_protocol_spread" in frame:
            fig.add_trace(
                go.Scatter(
                    x=frame["horizon"],
                    y=frame["median_protocol_spread"],
                    mode="lines+markers",
                    name="Median protocol spread",
                    line={"color": "#2563eb", "width": 3},
                    yaxis="y2",
                    hovertemplate="%{x}<br>Median spread %{y:.1%}<extra></extra>",
                )
            )
    fig.update_layout(
        barmode="stack",
        height=360,
        title="Ramp-protocol separation by horizon",
        xaxis_title="Entry horizon",
        yaxis={"title": "Share of strategies", "tickformat": ".0%"},
        yaxis2={
            "title": "Median return spread",
            "tickformat": ".0%",
            "overlaying": "y",
            "side": "right",
        },
        margin={"l": 20, "r": 20, "t": 48, "b": 90},
        legend={"orientation": "h", "yanchor": "top", "y": -0.25, "xanchor": "left", "x": 0.0},
    )
    return fig


def _launch_label_count(frame: pd.DataFrame, labels: set[str]) -> int:
    if frame.empty or not {"launch_label", "count"}.issubset(frame.columns):
        return 0
    filtered = frame[frame["launch_label"].astype(str).isin(labels)]
    return int(pd.to_numeric(filtered["count"], errors="coerce").fillna(0).sum())


def _transition_count(transitions: pd.DataFrame, direction: str) -> int:
    if transitions.empty or not {"direction", "count"}.issubset(transitions.columns):
        return 0
    filtered = transitions[transitions["direction"].astype(str).eq(direction)]
    return int(pd.to_numeric(filtered["count"], errors="coerce").fillna(0).sum())


def _transition_label_order(frame: pd.DataFrame) -> list[str]:
    labels = _launch_label_order()
    expected = [f"{left} -> {right}" for left in labels for right in labels]
    present = frame["transition"].dropna().astype(str).drop_duplicates().tolist()
    return [label for label in expected if label in present] + [
        label for label in present if label not in expected
    ]


def _launch_label_order() -> list[str]:
    return ["no_go", "wait", "set", "ready"]


def _launch_label_display(label: str) -> str:
    return str(label).replace("_", " ").title()


def _launch_label_colors() -> dict[str, str]:
    return {
        "no_go": "#dc2626",
        "wait": "#f59e0b",
        "set": "#2563eb",
        "ready": "#0f766e",
    }


def _render_launch_card_grid(cards: list[dict[str, str]], *, class_name: str) -> None:
    rendered_cards = []
    for card in cards:
        tone = html.escape(str(card.get("tone", "neutral")), quote=True)
        label = html.escape(str(card.get("label", "")), quote=True)
        answer = html.escape(str(card.get("answer", "")), quote=True)
        detail = html.escape(str(card.get("detail", "")), quote=True)
        rendered_cards.append(
            f'<div class="launch-guidance-card launch-guidance-{tone}">'
            f'<p class="launch-card-label">{label}</p>'
            f'<p class="launch-card-answer">{answer}</p>'
            f'<p class="launch-card-detail">{detail}</p>'
            "</div>"
        )
    st.markdown(
        f'<div class="{html.escape(class_name, quote=True)}">{"".join(rendered_cards)}</div>',
        unsafe_allow_html=True,
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return default
    return numeric


def _entry_scatter(windows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for protocol, frame in windows.groupby("protocol"):
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(frame["start_date"]),
                y=frame["total_return"],
                mode="markers",
                name=str(protocol),
                marker={
                    "size": frame["bad_start"].map({True: 12, False: 8}),
                    "opacity": 0.72,
                },
                customdata=frame[["excess_return", "max_drawdown", "bad_start"]],
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "Return %{y:.1%}<br>"
                    "Excess %{customdata[0]:.1%}<br>"
                    "Max DD %{customdata[1]:.1%}<br>"
                    "Bad start %{customdata[2]}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#64748b")
    fig.update_layout(
        height=360,
        title="Historical launch starts by protocol",
        xaxis_title="Start date",
        yaxis={"title": "Forward return", "tickformat": ".0%"},
        margin={"l": 20, "r": 20, "t": 48, "b": 76},
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "xanchor": "left", "x": 0.0},
    )
    return fig


def _protocol_bar(summary: pd.DataFrame, primary_horizon: str) -> go.Figure:
    frame = summary[summary["horizon"].astype(str).eq(primary_horizon)].copy()
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Bar(
                x=frame["protocol"],
                y=frame["launch_score"],
                name="Launch score",
                marker={"color": "#0f766e"},
                hovertemplate="%{x}<br>Launch score %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=frame["protocol"],
                y=frame["bad_start_rate"],
                name="Bad-start rate",
                marker={"color": "#dc2626"},
                hovertemplate="%{x}<br>Bad-start rate %{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=360,
        title=f"Launch protocol evidence: {primary_horizon}",
        yaxis={"title": "Score / rate", "tickformat": ".0%"},
        margin={"l": 20, "r": 20, "t": 48, "b": 108},
        legend={"orientation": "h", "yanchor": "top", "y": -0.32, "xanchor": "left", "x": 0.0},
    )
    return fig


def _ramp_figure(ramp_plan: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ramp_plan["week"],
            y=ramp_plan["capital_deployed"],
            mode="lines+markers",
            name="Capital deployed",
            line={"color": "#0f766e", "width": 3},
            hovertemplate="Week %{x}<br>Capital %{y:$,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ramp_plan["week"],
            y=ramp_plan["cash_reserved"],
            mode="lines+markers",
            name="Cash reserved",
            line={"color": "#f59e0b", "width": 3},
            hovertemplate="Week %{x}<br>Cash %{y:$,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        title="Launch ramp dollars",
        xaxis_title="Week",
        yaxis_title="Dollars",
        margin={"l": 20, "r": 20, "t": 48, "b": 72},
        legend={"orientation": "h", "yanchor": "top", "y": -0.24, "xanchor": "left", "x": 0.0},
    )
    return fig


def _launch_tone(label: str) -> str:
    normalized = label.lower()
    if normalized == "ready":
        return "success"
    if normalized == "set":
        return "warning"
    return "critical"
