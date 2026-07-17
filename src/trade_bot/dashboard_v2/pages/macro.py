from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_bot.dashboard.components import _render_metric_dataframe
from trade_bot.dashboard.formatting import _display_metrics
from trade_bot.dashboard.news_macro import (
    _current_event_rollup,
    _dedupe_display_rows,
    _driver_rotation_heatmap_figure,
    _driver_rotation_scatter_figure,
    _render_macro_driver_history,
    _render_news_and_macro,
)
from trade_bot.dashboard_v2.components.cards import (
    render_callout,
    render_card_grid,
    render_chart,
    render_section_header,
)
from trade_bot.dashboard_v2.services.runtime import DashboardRuntime
from trade_bot.DEFAULTS import (
    DEFAULT_NARRATIVE_OPERATING_DATA_SUPPORT,
    DEFAULT_NARRATIVE_RESEARCH_ONLY_DATA_SUPPORT,
    DEFAULT_NARRATIVE_UNSUPPORTED_DATA_SUPPORT,
)
from trade_bot.research.driver_rotation import (
    build_driver_rotation_table,
    summarize_driver_rotation,
)
from trade_bot.research.narrative_signals import (
    build_narrative_signal_table,
    summarize_narrative_signals,
)
from trade_bot.research.operating_history import _build_fast_current_state


_CURRENT_DRIVER_PLOT_LABEL = "Current snapshot"
_CUSTOM_DRIVER_PLOT_LABEL = "Custom date"
_MACRO_HISTORY_PRESETS: tuple[tuple[str, str], ...] = (
    ("Global Financial Crisis - lead-up", "2008-09-12"),
    ("Global Financial Crisis - unwind", "2009-03-09"),
    ("Global Financial Crisis - recovery", "2009-12-31"),
    ("Q4 2018 tightening - lead-up", "2018-09-28"),
    ("Q4 2018 tightening - unwind", "2018-12-24"),
    ("Q4 2018 tightening - recovery", "2019-06-30"),
    ("COVID liquidity crash - lead-up", "2020-02-18"),
    ("COVID liquidity crash - unwind", "2020-03-23"),
    ("COVID liquidity crash - recovery", "2020-08-31"),
    ("Inflation tech unwind - lead-up", "2021-12-31"),
    ("Inflation tech unwind - unwind", "2022-10-14"),
    ("Inflation tech unwind - recovery", "2023-07-31"),
)


@dataclass(frozen=True)
class _MacroDriverContext:
    as_of_date: pd.Timestamp | None
    current_state: Any
    driver_rotation: pd.DataFrame
    driver_summary: dict[str, str]
    narrative_summary: dict[str, str]
    is_current: bool
    requested_label: str


def render_macro_page(runtime: DashboardRuntime) -> None:
    current_state = runtime.baseline_run.current_state
    macro_data = runtime.baseline_run.macro_data
    prices = runtime.baseline_run.prices
    narrative_signals = build_narrative_signal_table(
        prices,
        news_triage=runtime.baseline_run.news_monitor.triage,
        events=runtime.baseline_run.event_risk.events,
    )
    narrative_summary = summarize_narrative_signals(narrative_signals)
    driver_rotation = build_driver_rotation_table(
        prices,
        current_state,
        narrative_signals=narrative_signals,
        news_triage=runtime.baseline_run.news_monitor.triage,
    )
    driver_summary = summarize_driver_rotation(driver_rotation)
    operating_signals = _signal_support(
        narrative_signals,
        DEFAULT_NARRATIVE_OPERATING_DATA_SUPPORT,
    )
    research_only_signals = _signal_support(
        narrative_signals,
        DEFAULT_NARRATIVE_RESEARCH_ONLY_DATA_SUPPORT,
    )
    unsupported_signals = _signal_support(
        narrative_signals,
        DEFAULT_NARRATIVE_UNSUPPORTED_DATA_SUPPORT,
    )
    render_card_grid(
        [
            ("Price Series", prices.shape[1]),
            ("Macro Series", macro_data.shape[1]),
            ("Macro Signals", len(current_state.macro_signals)),
            (
                "Pressure Groups",
                _macro_pressure_count(current_state.macro_category_summary),
            ),
            ("Active Drivers", _count_true(driver_rotation, "currently_active")),
            ("Emerging Drivers", _count_true(driver_rotation, "emerging_importance")),
            ("Fading Drivers", _count_true(driver_rotation, "fading_importance")),
            ("News Items", len(runtime.baseline_run.news_monitor.triage)),
            ("Operating Context", len(operating_signals)),
            (
                "Research-Only",
                len(research_only_signals) + len(unsupported_signals),
            ),
        ]
    )

    view = st.pills(
        "Macro view",
        ["Overview", "Visual Explorer", "Signal Drivers", "News & Events", "Full Workbench"],
        default="Overview",
        selection_mode="single",
        key="dashboard_v2_macro_view",
    )
    selected_view = view or "Overview"
    if selected_view == "Overview":
        _render_macro_overview(
            runtime,
            narrative_summary=narrative_summary,
            driver_rotation=driver_rotation,
            driver_summary=driver_summary,
        )
    elif selected_view == "Visual Explorer":
        _render_visual_explorer(prices=prices, macro_data=macro_data)
    elif selected_view == "Signal Drivers":
        _render_signal_drivers(
            runtime,
            driver_rotation=driver_rotation,
            driver_summary=driver_summary,
            narrative_signals=narrative_signals,
            narrative_summary=narrative_summary,
        )
    elif selected_view == "News & Events":
        _render_news_events(runtime, narrative_signals=narrative_signals)
    else:
        render_callout(
            "This loads the complete News & Macro workbench with narrative diagnostics, event triage, and raw tables.",
            heavy=True,
        )
        _render_news_and_macro(
            runtime.baseline_run,
            run_store_path=runtime.paths.run_store_path,
            artifact_dir=runtime.paths.artifact_dir,
            job_log_dir=runtime.paths.job_log_dir,
        )


def _render_macro_overview(
    runtime: DashboardRuntime,
    *,
    narrative_summary: dict[str, str],
    driver_rotation: pd.DataFrame,
    driver_summary: dict[str, str],
) -> None:
    render_section_header("Macro Overview")
    context = _render_macro_driver_time_controls(
        runtime,
        current_driver_rotation=driver_rotation,
        current_driver_summary=driver_summary,
        current_narrative_summary=narrative_summary,
    )
    current_state = context.current_state
    lead_regime = _lead_regime_read(current_state.growth_inflation_map)
    pressure_groups = _macro_pressure_count(current_state.macro_category_summary)
    top_driver = _top_driver(context.driver_rotation)
    if context.is_current:
        callout = (
            f"Lead macro regime is {lead_regime}. Driver rotation reads "
            f"{context.driver_summary['answer'].lower()} with {top_driver} most active. "
            f"Macro pressure groups: {pressure_groups}; narrative layer says "
            f"{context.narrative_summary['plain']}."
        )
    else:
        lead_clause = "" if lead_regime == "missing" else f" Lead macro regime was {lead_regime}."
        callout = (
            f"Historical read as of {context.as_of_date.date()}: driver rotation reads "
            f"{context.driver_summary['answer'].lower()} with {top_driver} most active."
            f"{lead_clause} This is reconstructed from local price/proxy data available through that date; "
            "today's news and event overlays are excluded."
        )
    render_callout(callout)
    if context.driver_rotation.empty:
        st.info("No driver-rotation diagnostics are available.")
    else:
        cols = st.columns(2)
        scatter, heatmap = _driver_rotation_figures_for_context(context)
        with cols[0]:
            render_chart(scatter)
        with cols[1]:
            render_chart(heatmap)
    _render_macro_driver_history(
        run_store_path=str(runtime.paths.run_store_path),
        artifact_dir=str(runtime.paths.artifact_dir),
        job_log_dir=str(runtime.paths.job_log_dir),
    )


def _render_macro_driver_time_controls(
    runtime: DashboardRuntime,
    *,
    current_driver_rotation: pd.DataFrame,
    current_driver_summary: dict[str, str],
    current_narrative_summary: dict[str, str],
) -> _MacroDriverContext:
    prices = runtime.baseline_run.prices
    options = _macro_history_preset_options(prices)
    control_cols = st.columns([2, 1])
    selected_label = control_cols[0].selectbox(
        "Driver plot date",
        options,
        index=0,
        key="dashboard_v2_macro_driver_plot_date",
    )
    selected_date: date | None = None
    if selected_label == _CUSTOM_DRIVER_PLOT_LABEL:
        available_dates = _available_price_dates(prices)
        latest_date = available_dates.max().date() if len(available_dates) else date.today()
        earliest_date = available_dates.min().date() if len(available_dates) else latest_date
        selected_date = control_cols[1].date_input(
            "As-of",
            value=latest_date,
            min_value=earliest_date,
            max_value=latest_date,
            key="dashboard_v2_macro_driver_custom_date",
        )
        requested_label = "Custom date"
    else:
        selected_date = _macro_history_preset_map(options).get(selected_label)
        requested_label = selected_label

    if selected_date is None:
        return _MacroDriverContext(
            as_of_date=None,
            current_state=runtime.baseline_run.current_state,
            driver_rotation=current_driver_rotation,
            driver_summary=current_driver_summary,
            narrative_summary=current_narrative_summary,
            is_current=True,
            requested_label=_CURRENT_DRIVER_PLOT_LABEL,
        )

    as_of_date = _market_date_on_or_before(prices, selected_date)
    if as_of_date is None:
        st.warning("No price history is available on or before the selected date.")
        return _MacroDriverContext(
            as_of_date=None,
            current_state=runtime.baseline_run.current_state,
            driver_rotation=current_driver_rotation,
            driver_summary=current_driver_summary,
            narrative_summary=current_narrative_summary,
            is_current=True,
            requested_label=_CURRENT_DRIVER_PLOT_LABEL,
        )
    if as_of_date.date() != selected_date:
        st.caption(f"Using nearest available market date: {as_of_date.date()}.")

    historical_prices = _prices_as_of(prices, as_of_date)
    if len(historical_prices.dropna(how="all")) < 252:
        st.warning("Historical reconstruction needs at least 252 trading days before the selected date.")
    historical_state = _build_fast_current_state(historical_prices)
    historical_narrative = build_narrative_signal_table(
        historical_prices,
        news_triage=pd.DataFrame(),
        events=(),
    )
    historical_rotation = build_driver_rotation_table(
        historical_prices,
        historical_state,
        narrative_signals=historical_narrative,
        news_triage=pd.DataFrame(),
    )
    return _MacroDriverContext(
        as_of_date=as_of_date,
        current_state=historical_state,
        driver_rotation=historical_rotation,
        driver_summary=summarize_driver_rotation(historical_rotation),
        narrative_summary=summarize_narrative_signals(historical_narrative),
        is_current=False,
        requested_label=requested_label,
    )


def _driver_rotation_figures_for_context(
    context: _MacroDriverContext,
) -> tuple[go.Figure, go.Figure]:
    scatter = _driver_rotation_scatter_figure(context.driver_rotation)
    heatmap = _driver_rotation_heatmap_figure(context.driver_rotation)
    if context.is_current or context.as_of_date is None:
        return scatter, heatmap
    date_label = str(context.as_of_date.date())
    scatter.update_layout(
        title=f"Driver Rotation as of {date_label}: Historical Relevance vs Activation",
    )
    scatter.update_yaxes(title_text="Activation / pressure as of selected date")
    if heatmap.data:
        heatmap.data[0].x = [
            "90d proxy activation",
            "30d proxy activation",
            "Selected-date activation",
            "Historical relevance",
        ]
    heatmap.update_layout(title=f"Driver Activation Heatmap as of {date_label}")
    return scatter, heatmap


def _macro_history_preset_options(prices: pd.DataFrame) -> list[str]:
    options = [_CURRENT_DRIVER_PLOT_LABEL]
    available_dates = _available_price_dates(prices)
    if len(available_dates):
        earliest = available_dates.min().date()
        latest = available_dates.max().date()
        for label, raw_date in _MACRO_HISTORY_PRESETS:
            preset_date = date.fromisoformat(raw_date)
            if earliest <= preset_date <= latest:
                options.append(f"{label} ({raw_date})")
    options.append(_CUSTOM_DRIVER_PLOT_LABEL)
    return options


def _macro_history_preset_map(options: list[str]) -> dict[str, date | None]:
    mapping: dict[str, date | None] = {_CURRENT_DRIVER_PLOT_LABEL: None}
    for option in options:
        if option in {_CURRENT_DRIVER_PLOT_LABEL, _CUSTOM_DRIVER_PLOT_LABEL}:
            continue
        raw_date = option.rsplit("(", 1)[-1].rstrip(")")
        try:
            mapping[option] = date.fromisoformat(raw_date)
        except ValueError:
            continue
    return mapping


def _available_price_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    if prices.empty:
        return pd.DatetimeIndex([])
    dates = pd.to_datetime(prices.index, errors="coerce")
    dates = pd.DatetimeIndex(dates).dropna().sort_values().unique()
    return pd.DatetimeIndex(dates)


def _market_date_on_or_before(prices: pd.DataFrame, requested: date) -> pd.Timestamp | None:
    dates = _available_price_dates(prices)
    if len(dates) == 0:
        return None
    requested_ts = pd.Timestamp(requested)
    prior_dates = dates[dates <= requested_ts]
    if len(prior_dates) == 0:
        return None
    return pd.Timestamp(prior_dates.max())


def _prices_as_of(prices: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if prices.empty:
        return prices
    data = prices.copy()
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[~pd.isna(data.index)].sort_index()
    return data.loc[:as_of_date].copy()


def _render_visual_explorer(*, prices: pd.DataFrame, macro_data: pd.DataFrame) -> None:
    render_section_header("Macro Visual Explorer")
    st.caption(
        "Use this to inspect raw market proxies and macro series before drilling into narrative tables."
    )
    range_cols = st.columns([1, 1, 2])
    range_choice = range_cols[0].selectbox(
        "Time range",
        ["6M", "1Y", "3Y", "5Y", "All", "Custom"],
        index=2,
        key="dashboard_v2_macro_range",
    )
    custom_start = None
    custom_end = None
    if range_choice == "Custom":
        custom_start = range_cols[1].date_input(
            "Start",
            value=date.today() - timedelta(days=365),
            key="dashboard_v2_macro_start",
        )
        custom_end = range_cols[2].date_input(
            "End",
            value=date.today(),
            key="dashboard_v2_macro_end",
        )

    price_options = list(prices.columns.astype(str)) if not prices.empty else []
    default_tickers = [ticker for ticker in ["SPY", "QQQ", "BIL", "VEA", "IWM", "GLD", "TLT"] if ticker in price_options]
    selected_tickers = st.multiselect(
        "Market tickers",
        price_options,
        default=default_tickers[:6] or price_options[:4],
        key="dashboard_v2_macro_price_tickers",
    )
    price_frame = _slice_time_range(
        prices[selected_tickers] if selected_tickers and not prices.empty else pd.DataFrame(),
        range_choice,
        custom_start=custom_start,
        custom_end=custom_end,
    )
    price_figure = _indexed_price_figure(price_frame, title="Selected Market Proxies")
    if price_figure is None:
        st.info("No selected price series are available.")
    else:
        render_chart(price_figure)

    numeric_macro = macro_data.select_dtypes(include="number") if not macro_data.empty else pd.DataFrame()
    macro_options = list(numeric_macro.columns.astype(str))
    macro_defaults = macro_options[:5]
    macro_cols = st.columns([3, 1])
    selected_macro = macro_cols[0].multiselect(
        "Macro series",
        macro_options,
        default=macro_defaults,
        key="dashboard_v2_macro_series",
    )
    transform = macro_cols[1].selectbox(
        "Macro scale",
        ["Z-score", "Level"],
        index=0,
        key="dashboard_v2_macro_scale",
    )
    macro_frame = _slice_time_range(
        numeric_macro[selected_macro] if selected_macro and not numeric_macro.empty else pd.DataFrame(),
        range_choice,
        custom_start=custom_start,
        custom_end=custom_end,
    )
    macro_figure = _macro_signal_figure(macro_frame, title="Selected Macro Signals", zscore=transform == "Z-score")
    if macro_figure is None:
        st.info("No selected macro series are available.")
    else:
        render_chart(macro_figure)


def _render_signal_drivers(
    runtime: DashboardRuntime,
    *,
    driver_rotation: pd.DataFrame,
    driver_summary: dict[str, str],
    narrative_signals: pd.DataFrame,
    narrative_summary: dict[str, str],
) -> None:
    current_state = runtime.baseline_run.current_state
    render_section_header("Signal Drivers")
    render_callout(
        
            f"{driver_summary['detail']} {narrative_summary['detail']} "
            "Use this page to inspect whether pressure is broad, emerging, fading, or only explanatory."
        
    )
    if not driver_rotation.empty:
        driver_cols = [
            "driver_label",
            "model_role",
            "primary_rotation_state",
            "proven_relevance",
            "current_activation",
            "previous_30d_activation",
            "previous_90d_activation",
            "change_30d",
            "change_90d",
            "data_support",
            "evidence",
            "interpretation",
        ]
        _render_metric_dataframe(
            _display_metrics(driver_rotation[[col for col in driver_cols if col in driver_rotation]]),
            hide_index=True,
        )
    else:
        st.info("No driver-rotation diagnostics are available.")
    if current_state.macro_category_summary.empty:
        st.info("No macro category summary is available.")
    else:
        render_section_header("Category Summary")
        _render_metric_dataframe(_display_metrics(current_state.macro_category_summary))
    if current_state.macro_signals.empty:
        st.info("No macro signal table is available.")
    else:
        render_section_header("Signal Detail")
        _render_metric_dataframe(_display_metrics(current_state.macro_signals))
    if not narrative_signals.empty:
        render_section_header("Cross-Source Insight Diagnostics")
        _render_metric_dataframe(_display_metrics(narrative_signals), hide_index=True)


def _render_news_events(
    runtime: DashboardRuntime,
    *,
    narrative_signals: pd.DataFrame,
) -> None:
    render_section_header("News & Events")
    triage = runtime.baseline_run.news_monitor.triage
    current_event_scenarios = runtime.baseline_run.event_risk.current_event_scenarios
    if triage.empty and current_event_scenarios.empty:
        st.info("No current news or event-risk records are available.")
        return
    if not triage.empty:
        activation_options = ["all", *sorted(triage["activation_status"].dropna().unique())]
        category_options = ["all", *sorted(triage["category"].dropna().unique())]
        filter_cols = st.columns(2)
        activation = filter_cols[0].selectbox(
            "Activation",
            activation_options,
            key="dashboard_v2_macro_news_activation",
        )
        category = filter_cols[1].selectbox(
            "Category",
            category_options,
            key="dashboard_v2_macro_news_category",
        )
        news_view = triage.copy()
        if activation != "all":
            news_view = news_view[news_view["activation_status"] == activation]
        if category != "all":
            news_view = news_view[news_view["category"] == category]
        news_columns = [
            "title",
            "source",
            "published_at",
            "category",
            "direction",
            "phase",
            "urgency_score",
            "activation_status",
            "event_id",
            "risk_channels",
            "candidate_proxies",
            "confirmation_window",
            "url",
        ]
        render_section_header("Current News/Event Triage")
        _render_metric_dataframe(
            _display_metrics(_dedupe_display_rows(news_view, news_columns).head(100))
        )
    if not current_event_scenarios.empty:
        render_section_header("Event-Risk Monitor")
        current_event_rollup = _current_event_rollup(current_event_scenarios)
        _render_metric_dataframe(_display_metrics(current_event_rollup), hide_index=True)
    unsupported = _signal_support(
        narrative_signals,
        DEFAULT_NARRATIVE_UNSUPPORTED_DATA_SUPPORT,
    )
    if not unsupported.empty:
        render_section_header("Unsupported Narrative Watchlist")
        render_callout(
            "These are data gaps or watchlist themes. They can explain the narrative, but do not have allocation authority."
        )
        _render_metric_dataframe(_display_metrics(unsupported), hide_index=True)


def _slice_time_range(
    frame: pd.DataFrame,
    range_choice: str,
    *,
    custom_start: date | None,
    custom_end: date | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    working.index = pd.to_datetime(working.index, errors="coerce")
    working = working[working.index.notna()].sort_index()
    if working.empty or range_choice == "All":
        return working
    end = pd.to_datetime(custom_end) if custom_end is not None else working.index.max()
    if range_choice == "Custom":
        start = pd.to_datetime(custom_start) if custom_start is not None else working.index.min()
    else:
        days = {"6M": 183, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}.get(range_choice, 365 * 3)
        start = end - pd.Timedelta(days=days)
    return working[(working.index >= start) & (working.index <= end)]


def _indexed_price_figure(frame: pd.DataFrame, *, title: str) -> go.Figure | None:
    if frame.empty:
        return None
    normalized = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if normalized.empty:
        return None
    normalized = normalized.ffill()
    base = normalized.apply(lambda series: series.dropna().iloc[0] if not series.dropna().empty else pd.NA)
    indexed = normalized.divide(base.replace(0, pd.NA), axis=1) - 1.0
    fig = go.Figure()
    for column in indexed.columns:
        fig.add_trace(
            go.Scatter(
                x=indexed.index,
                y=indexed[column],
                mode="lines",
                name=str(column),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        height=390,
        yaxis={"title": "Indexed return", "tickformat": ".0%"},
        margin={"l": 20, "r": 20, "t": 55, "b": 50},
        legend={"orientation": "h", "y": -0.20},
    )
    return fig


def _macro_signal_figure(frame: pd.DataFrame, *, title: str, zscore: bool) -> go.Figure | None:
    if frame.empty:
        return None
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all").ffill()
    if numeric.empty:
        return None
    display = numeric
    y_title = "Level"
    hover = "%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>"
    if zscore:
        display = numeric.apply(_zscore)
        y_title = "Z-score"
        hover = "%{x|%Y-%m-%d}<br>%{y:.2f} z<extra></extra>"
    fig = go.Figure()
    for column in display.columns:
        fig.add_trace(
            go.Scatter(
                x=display.index,
                y=display[column],
                mode="lines",
                name=str(column),
                hovertemplate=hover,
            )
        )
    fig.update_layout(
        title=title,
        height=390,
        yaxis_title=y_title,
        margin={"l": 20, "r": 20, "t": 55, "b": 50},
        legend={"orientation": "h", "y": -0.20},
    )
    return fig


def _zscore(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    std = float(clean.std())
    if std == 0.0 or pd.isna(std):
        return clean * 0.0
    return (clean - float(clean.mean())) / std


def _macro_pressure_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    state_columns = [
        column
        for column in ("risk_state", "near_term_state", "latest_pressure_state", "state")
        if column in frame
    ]
    if not state_columns:
        return 0
    return int(frame[state_columns[0]].astype(str).str.contains("risk_pressure").sum())


def _signal_support(frame: pd.DataFrame, support_values: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty or "data_support" not in frame:
        return pd.DataFrame()
    return frame[frame["data_support"].isin(support_values)]


def _count_true(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _lead_regime_read(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "missing"
    row = frame.iloc[0]
    regime = str(row.get("regime", "missing"))
    probability = row.get("probability")
    try:
        return f"{regime} ({float(probability):.0%})"
    except (TypeError, ValueError):
        return regime


def _top_driver(driver_rotation: pd.DataFrame) -> str:
    if driver_rotation.empty:
        return "no driver"
    frame = driver_rotation.sort_values(
        ["current_activation", "proven_relevance"],
        ascending=False,
    )
    row = frame.iloc[0]
    label = str(row.get("driver_label", row.get("driver", "unknown driver")))
    activation = row.get("current_activation")
    try:
        return f"{label} ({float(activation):.0%})"
    except (TypeError, ValueError):
        return label
