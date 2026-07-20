from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics, metrics_frame
from trade_bot.backtest.windows import (
    calendar_return_pivot,
    calendar_year_metrics,
    rolling_window_metrics,
    summarize_windows,
)
from trade_bot.config import BotConfig, configured_tickers
from trade_bot.data.fred_data import FredSeries, load_fred_catalog, load_or_fetch_fred_data
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.DEFAULTS import DEFAULT_EVENTS_PATH, DEFAULT_MACRO_PATH, DEFAULT_NEWS_PATH
from trade_bot.portfolio.risk import PortfolioRiskRun
from trade_bot.research.current_state import CurrentStateRun, build_current_state
from trade_bot.research.event_risk import (
    EventRiskRun,
    MarketEvent,
    load_market_events,
    run_event_risk_study,
)
from trade_bot.research.news_monitor import (
    NewsMonitorRun,
    activate_news_events,
    run_news_monitor,
)
from trade_bot.research.signal_inclusion import SignalInclusionRun, run_signal_inclusion_tests
from trade_bot.research.trade_decision import TradeDecisionRun, build_trade_decision
from trade_bot.strategies.momentum import build_strategy_weights


@dataclass(frozen=True)
class BaselineRun:
    prices: pd.DataFrame
    macro_data: pd.DataFrame
    macro_catalog: tuple[FredSeries, ...]
    results: dict[str, BacktestResult]
    metrics: pd.DataFrame
    rolling_windows: pd.DataFrame
    window_summary: pd.DataFrame
    calendar_metrics: pd.DataFrame
    calendar_returns: pd.DataFrame
    current_state: CurrentStateRun
    event_risk: EventRiskRun
    news_monitor: NewsMonitorRun
    signal_inclusion: SignalInclusionRun
    trade_decision: TradeDecisionRun
    portfolio_risk: PortfolioRiskRun | None = None


def run_configured_baselines(
    config: BotConfig,
    *,
    refresh_data: bool = False,
    refresh_macro: bool = False,
    refresh_news: bool = False,
    event_config_path: str | Path | None = DEFAULT_EVENTS_PATH,
    macro_config_path: str | Path | None = DEFAULT_MACRO_PATH,
    news_config_path: str | Path | None = DEFAULT_NEWS_PATH,
    as_of: str | pd.Timestamp | None = None,
) -> BaselineRun:
    prices = load_or_fetch_yahoo_prices(
        configured_tickers(config),
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    )
    macro_catalog = load_fred_catalog(macro_config_path)
    macro_data = load_or_fetch_fred_data(
        macro_catalog,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        refresh=refresh_macro,
    )
    return run_configured_baselines_from_frames(
        config,
        prices=prices,
        macro_data=macro_data,
        macro_catalog=macro_catalog,
        refresh_news=refresh_news,
        event_config_path=event_config_path,
        news_config_path=news_config_path,
        as_of=as_of,
    )


def run_configured_baselines_from_frames(
    config: BotConfig,
    *,
    prices: pd.DataFrame,
    macro_data: pd.DataFrame | None = None,
    macro_catalog: tuple[FredSeries, ...] = (),
    refresh_news: bool = False,
    event_config_path: str | Path | None = DEFAULT_EVENTS_PATH,
    news_config_path: str | Path | None = DEFAULT_NEWS_PATH,
    as_of: str | pd.Timestamp | None = None,
) -> BaselineRun:
    as_of_utc = _as_of_timestamp(as_of)
    prices = prices.dropna(how="all").sort_index()
    macro_data = (
        macro_data.dropna(how="all").sort_index()
        if isinstance(macro_data, pd.DataFrame)
        else pd.DataFrame()
    )

    results: dict[str, BacktestResult] = {}
    calculated_metrics: list[PerformanceMetrics] = []
    for name, strategy in config.strategies.items():
        strategy_prices = _strategy_prices(prices, strategy.tickers, strategy.defensive_ticker)
        if strategy_prices.empty:
            raise RuntimeError(
                f"Strategy {name!r} has no usable price rows for tickers "
                f"{strategy.tickers!r} and defensive ticker {strategy.defensive_ticker!r}."
            )
        target_weights = build_strategy_weights(strategy_prices, strategy)
        result = run_backtest(
            name,
            strategy_prices,
            target_weights,
            config.execution,
            volatility_target=strategy.volatility_target,
            drawdown_control=strategy.drawdown_control,
        )
        if result.returns.dropna().empty:
            raise RuntimeError(
                f"Strategy {name!r} produced an empty return series. "
                "Check for all-empty price columns after the latest data refresh."
            )
        results[name] = result
        calculated_metrics.append(
            calculate_metrics(
                name=result.name,
                returns=result.returns,
                equity=result.equity,
                turnover=result.turnover,
                transaction_costs=result.transaction_costs,
            )
        )

    rolling_windows = rolling_window_metrics(results)
    calendar_metrics_frame = calendar_year_metrics(results)
    current_state = build_current_state(
        prices,
        results,
        macro_data=macro_data,
        macro_catalog=macro_catalog,
    )
    events = _events_as_of(load_market_events(event_config_path), as_of_utc)
    news_monitor = run_news_monitor(
        news_config_path,
        cache_dir=config.data.cache_dir,
        refresh=refresh_news,
        now=as_of_utc,
    )
    news_monitor = activate_news_events(news_monitor, events)
    event_risk = run_event_risk_study(prices, results, (*events, *news_monitor.activated_events))
    primary_strategy = config.primary_strategy
    if primary_strategy not in results:
        msg = f"Configured primary strategy {primary_strategy!r} was not found in strategies."
        raise KeyError(msg)
    signal_inclusion = run_signal_inclusion_tests(
        prices,
        macro_data,
        macro_catalog,
        results[primary_strategy],
        config.execution,
        base_strategy_name=primary_strategy,
    )
    trade_decision = build_trade_decision(
        primary_result=results[primary_strategy],
        current_state=current_state,
        event_risk=event_risk,
        news_monitor=news_monitor,
        signal_inclusion=signal_inclusion,
        prices=prices,
    )

    return BaselineRun(
        prices=prices,
        macro_data=macro_data,
        macro_catalog=macro_catalog,
        results=results,
        metrics=metrics_frame(calculated_metrics).sort_values("calmar", ascending=False),
        rolling_windows=rolling_windows,
        window_summary=summarize_windows(rolling_windows),
        calendar_metrics=calendar_metrics_frame,
        calendar_returns=calendar_return_pivot(calendar_metrics_frame),
        current_state=current_state,
        event_risk=event_risk,
        news_monitor=news_monitor,
        signal_inclusion=signal_inclusion,
        trade_decision=trade_decision,
        portfolio_risk=trade_decision.portfolio_risk,
    )


def _strategy_prices(
    prices: pd.DataFrame,
    tickers: list[str],
    defensive_ticker: str | None,
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*tickers, *([defensive_ticker] if defensive_ticker else [])]))
    available = prices[columns].dropna(how="all")
    return available


def _as_of_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _events_as_of(
    events: tuple[MarketEvent, ...],
    as_of_utc: pd.Timestamp | None,
) -> tuple[MarketEvent, ...]:
    if as_of_utc is None:
        return events
    cutoff_date = as_of_utc.tz_convert("UTC").date()
    return tuple(event for event in events if _event_date(event) <= cutoff_date)


def _event_date(event: MarketEvent) -> object:
    timestamp = pd.Timestamp(event.date)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.date()
