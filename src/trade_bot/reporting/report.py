from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trade_bot.backtest.engine import BacktestResult
from trade_bot.features.indicators import TRADING_DAYS_PER_YEAR, drawdown
from trade_bot.portfolio.risk import current_positions
from trade_bot.research.current_state import CurrentStateRun
from trade_bot.research.event_risk import EventRiskRun
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.signal_inclusion import SignalInclusionRun
from trade_bot.research.trade_decision import TradeDecisionRun


def write_baseline_report(
    results: dict[str, BacktestResult],
    metrics: pd.DataFrame,
    window_summary: pd.DataFrame | None,
    calendar_returns: pd.DataFrame | None,
    current_state: CurrentStateRun | None,
    event_risk: EventRiskRun | None,
    news_monitor: NewsMonitorRun | None,
    signal_inclusion: SignalInclusionRun | None,
    trade_decision: TradeDecisionRun | None,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    html = "\n".join(
        [
            "<html><head><title>Trade Bot Baseline Report</title></head><body>",
            "<h1>Trade Bot Baseline Report</h1>",
            "<p>Backtests assume next-session execution, long-only weights, and explicit turnover costs.</p>",
            "<h2>Trade Decision</h2>",
            _trade_decision_html(trade_decision),
            "<h2>Current State</h2>",
            _current_state_html(current_state),
            "<h2>Performance Metrics</h2>",
            _format_metrics(metrics).to_html(float_format=lambda x: f"{x:,.4f}"),
            "<h2>Rolling Window Summary</h2>",
            _optional_table(window_summary),
            "<h2>Calendar Year Returns</h2>",
            _optional_percent_table(calendar_returns),
            "<h2>Event-Risk Scenarios</h2>",
            _news_monitor_html(news_monitor),
            _event_risk_html(event_risk),
            "<h2>Signal Inclusion Tests</h2>",
            _signal_inclusion_html(signal_inclusion),
            "<h2>Equity and Drawdown</h2>",
            make_equity_drawdown_figure(results).to_html(full_html=False, include_plotlyjs="cdn"),
            "<h2>Latest Positions</h2>",
            latest_positions_frame(results).to_html(float_format=lambda x: f"{x:,.2%}"),
            "</body></html>",
        ]
    )
    output.write_text(html, encoding="utf-8")
    return output


def make_equity_drawdown_figure(
    results: dict[str, BacktestResult],
    *,
    strategy_names: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    rebase: bool = True,
    title: str | None = None,
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Growth of $1", "Drawdown"),
    )

    selected_results = _selected_results(results, strategy_names)
    for name, result in selected_results.items():
        equity = _windowed_series(result.equity, start=start, end=end)
        if equity.empty:
            continue
        normalized = equity / equity.iloc[0] if rebase else equity
        figure.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized,
                mode="lines",
                name=name,
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.4f}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )
        dd = drawdown(normalized)
        figure.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd,
                mode="lines",
                name=f"{name} drawdown",
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra>%{fullData.name}</extra>",
            ),
            row=2,
            col=1,
        )

    figure.update_yaxes(tickprefix="$", tickformat=".2f", row=1, col=1)
    figure.update_yaxes(tickformat=".0%", row=2, col=1)
    figure.update_layout(
        template="plotly_white",
        height=850,
        hovermode="x unified",
        title=title,
    )
    return figure


def window_performance_frame(
    results: dict[str, BacktestResult],
    *,
    strategy_names: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows = []
    for name, result in _selected_results(results, strategy_names).items():
        equity = _windowed_series(result.equity, start=start, end=end)
        if equity.empty:
            continue
        normalized = equity / equity.iloc[0]
        window_drawdown = drawdown(normalized)
        returns = normalized.pct_change(fill_method=None).fillna(0.0)
        years = _window_years(normalized)
        total_return = float(normalized.iloc[-1] - 1.0)
        cagr = _window_cagr(normalized, years)
        annualized_volatility = _annualized_volatility(returns)
        sharpe = _safe_ratio(float(returns.mean()) * TRADING_DAYS_PER_YEAR, annualized_volatility)
        max_drawdown = float(window_drawdown.min())
        turnover = result.turnover.reindex(equity.index).fillna(0.0).copy()
        turnover.iloc[0] = 0.0
        rows.append(
            {
                "strategy": name,
                "start": str(equity.index.min().date()),
                "end": str(equity.index.max().date()),
                "observations": int(equity.shape[0]),
                "years": years,
                "total_return": total_return,
                "cagr": cagr,
                "annualized_volatility": annualized_volatility,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "current_drawdown": float(window_drawdown.iloc[-1]),
                "calmar": _safe_ratio(cagr, abs(max_drawdown)),
                "best_day": float(returns.max()),
                "worst_day": float(returns.min()),
                "average_turnover": float(turnover.mean()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("total_return", ascending=False)


def latest_positions_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    for name, result in results.items():
        rows[name] = current_positions(result.weights)
    return pd.DataFrame(rows).fillna(0.0).sort_index()


def _selected_results(
    results: dict[str, BacktestResult],
    strategy_names: list[str] | None,
) -> dict[str, BacktestResult]:
    if strategy_names is None:
        return results
    selected = set(strategy_names)
    return {name: result for name, result in results.items() if name in selected}


def _windowed_series(
    series: pd.Series,
    *,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.Series:
    windowed = series.sort_index().dropna()
    if start is not None:
        windowed = windowed.loc[windowed.index >= pd.Timestamp(start)]
    if end is not None:
        windowed = windowed.loc[windowed.index <= pd.Timestamp(end)]
    return windowed


def _window_years(equity: pd.Series) -> float:
    elapsed_days = (equity.index[-1] - equity.index[0]).days
    return max(elapsed_days / 365.25, 1 / 365.25)


def _window_cagr(normalized_equity: pd.Series, years: float) -> float:
    final_growth = float(normalized_equity.iloc[-1])
    if final_growth <= 0.0:
        return -1.0
    return float(final_growth ** (1.0 / years) - 1.0)


def _annualized_volatility(returns: pd.Series) -> float:
    if returns.shape[0] < 2:
        return 0.0
    volatility = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    if np.isnan(volatility):
        return 0.0
    return volatility


def _safe_ratio(numerator: float, denominator: float) -> float:
    if np.isnan(numerator) or np.isnan(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()
    percent_columns = [
        "cagr",
        "annualized_volatility",
        "max_drawdown",
        "best_day",
        "worst_day",
        "average_turnover",
        "total_transaction_cost",
    ]
    for column in percent_columns:
        if column in formatted:
            formatted[column] = formatted[column].astype(float)
    return formatted


def _optional_table(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "<p>No window diagnostics available.</p>"
    return frame.to_html(float_format=lambda x: f"{x:,.4f}")


def _optional_percent_table(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "<p>No calendar diagnostics available.</p>"
    return frame.to_html(float_format=lambda x: f"{x:,.2%}")


def _current_state_html(current_state: CurrentStateRun | None) -> str:
    if current_state is None:
        return "<p>No current-state diagnostics available.</p>"
    return "\n".join(
        [
            f"<p><strong>Date:</strong> {current_state.market_date}</p>",
            f"<p><strong>Risk Status:</strong> {current_state.risk_status.upper()} "
            f"({current_state.risk_score:.2f})</p>",
            f"<p>{current_state.risk_summary}</p>",
            "<h3>Strategy Alerts</h3>",
            current_state.strategy_alerts.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            "<h3>Signal Coverage</h3>",
            current_state.signal_coverage.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            "<h3>Regime Pulse Cycles</h3>",
            current_state.regime_pulse_cycles.to_html(
                index=False, float_format=lambda x: f"{x:,.4f}"
            ),
            "<h3>Regime Pulse Asset Reads</h3>",
            current_state.regime_pulse_assets.to_html(
                index=False, float_format=lambda x: f"{x:,.4f}"
            ),
            "<h3>Growth-Inflation Map Probabilities</h3>",
            current_state.growth_inflation_map.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            "<h3>Positioning And Crowding Summary</h3>",
            current_state.positioning_summary.to_html(
                index=False, float_format=lambda x: f"{x:,.4f}"
            ),
            "<h3>Macro Category Summary</h3>",
            current_state.macro_category_summary.to_html(
                index=False, float_format=lambda x: f"{x:,.4f}"
            ),
            "<h3>Macro Signals</h3>",
            current_state.macro_signals.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            "<h3>Scenario Outlook</h3>",
            current_state.scenario_outlook.to_html(index=False, float_format=lambda x: f"{x:,.2%}"),
            "<h3>Scenario Drivers</h3>",
            current_state.scenario_drivers.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            "<h3>Scenario Lattice</h3>",
            current_state.scenario_lattice.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            "<h3>Risk Confirmation Matrix</h3>",
            current_state.confirmation_matrix.to_html(
                index=False, float_format=lambda x: f"{x:,.4f}"
            ),
            "<h3>Market Health</h3>",
            current_state.market_health.to_html(float_format=lambda x: f"{x:,.4f}"),
        ]
    )


def _event_risk_html(event_risk: EventRiskRun | None) -> str:
    if event_risk is None or not event_risk.events:
        return "<p>No event-risk diagnostics available.</p>"

    sections = [
        "<p>Event windows use close-to-close returns around curated policy and geopolitical shocks. "
        "Incomplete future windows are left blank.</p>",
    ]
    if not event_risk.current_event_scenarios.empty:
        sections.extend(
            [
                "<h3>Current Event Scenario Playbook</h3>",
                event_risk.current_event_scenarios.to_html(
                    index=False,
                    float_format=lambda x: f"{x:,.4f}",
                ),
            ]
        )
    if not event_risk.event_summary.empty:
        sections.extend(
            [
                "<h3>Historical Event-Window Summary</h3>",
                event_risk.event_summary.to_html(index=False, float_format=lambda x: f"{x:,.2%}"),
            ]
        )
    if not event_risk.strategy_event_returns.empty:
        strategy_returns = event_risk.strategy_event_returns[
            event_risk.strategy_event_returns["complete"]
        ].copy()
        sections.extend(
            [
                "<h3>Strategy Event Returns</h3>",
                strategy_returns.to_html(index=False, float_format=lambda x: f"{x:,.2%}"),
            ]
        )
    return "\n".join(sections)


def _news_monitor_html(news_monitor: NewsMonitorRun | None) -> str:
    if news_monitor is None:
        return "<p>No news-intake diagnostics available.</p>"

    sections = [
        "<h3>News Intake Monitor</h3>",
        "<p>High-urgency classified news can generate current event-risk scenarios. "
        "Curated duplicates are marked as already covered.</p>",
    ]
    if not news_monitor.source_health.empty:
        sections.extend(
            [
                "<h4>Source Health</h4>",
                news_monitor.source_health.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            ]
        )
    if not news_monitor.triage.empty:
        display_columns = [
            "title",
            "source",
            "published_at",
            "category",
            "direction",
            "phase",
            "urgency_score",
            "activation_status",
            "event_id",
            "candidate_proxies",
            "confirmation_window",
            "url",
        ]
        available_columns = [
            column for column in display_columns if column in news_monitor.triage.columns
        ]
        sections.extend(
            [
                "<h4>News Triage</h4>",
                news_monitor.triage[available_columns]
                .head(50)
                .to_html(
                    index=False,
                    float_format=lambda x: f"{x:,.4f}",
                ),
            ]
        )
    else:
        sections.append("<p>No recent news items were triaged.</p>")
    return "\n".join(sections)


def _signal_inclusion_html(signal_inclusion: SignalInclusionRun | None) -> str:
    if signal_inclusion is None or signal_inclusion.summary.empty:
        return "<p>No signal-inclusion diagnostics available.</p>"

    display_columns = [
        "signal_group",
        "test_status",
        "decision",
        "latest_pressure_state",
        "latest_pressure",
        "active_day_rate",
        "delta_cagr",
        "delta_sharpe",
        "max_drawdown_improvement",
        "delta_calmar",
        "delta_worst_3y_cagr",
        "revision_safe",
        "rationale",
    ]
    available_columns = [
        column for column in display_columns if column in signal_inclusion.summary.columns
    ]
    return "\n".join(
        [
            "<p>Macro inclusion tests apply a lagged, risk-reduction-only overlay to the "
            "primary strategy. These are exploratory because FRED observation-date histories "
            "are not revision-safe.</p>",
            signal_inclusion.summary[available_columns].to_html(
                index=False,
                float_format=lambda x: f"{x:,.4f}",
            ),
        ]
    )


def _trade_decision_html(trade_decision: TradeDecisionRun | None) -> str:
    if trade_decision is None or trade_decision.summary.empty:
        return "<p>No trade-decision diagnostics available.</p>"

    sections = [
        "<p>The trade decision connects systematic holdings to scenario, event, news, and "
        "validated-signal context. Scenario-adjusted weights are review targets, not automatic "
        "execution instructions.</p>",
        "<h3>Recommendation</h3>",
        trade_decision.summary.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
    ]
    if not trade_decision.position_plan.empty:
        sections.extend(
            [
                "<h3>Position-Sizing Bridge</h3>",
                trade_decision.position_plan.to_html(
                    index=False,
                    float_format=lambda x: f"{x:,.4f}",
                ),
            ]
        )
    if not trade_decision.evidence.empty:
        sections.extend(
            [
                "<h3>Decision Evidence</h3>",
                trade_decision.evidence.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            ]
        )
    if not trade_decision.scenario_links.empty:
        sections.extend(
            [
                "<h3>Scenario Links</h3>",
                trade_decision.scenario_links.to_html(
                    index=False,
                    float_format=lambda x: f"{x:,.4f}",
                ),
            ]
        )
    if (
        trade_decision.portfolio_risk is not None
        and not trade_decision.portfolio_risk.summary.empty
    ):
        risk = trade_decision.portfolio_risk
        sections.extend(
            [
                "<h3>Portfolio Risk Engine</h3>",
                risk.summary.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
                "<h4>Constraint Report</h4>",
                risk.constraint_report.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
                "<h4>Factor Exposures</h4>",
                risk.factor_exposures.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
                "<h4>Stress Tests</h4>",
                risk.stress_tests.to_html(index=False, float_format=lambda x: f"{x:,.4f}"),
            ]
        )
    return "\n".join(sections)
