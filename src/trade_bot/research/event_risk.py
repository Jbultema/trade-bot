from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import yaml

from trade_bot.backtest.engine import BacktestResult
from trade_bot.DEFAULT import DEFAULT_EVENT_ASSET_PROXIES, DEFAULT_EVENT_WINDOWS

EventDirection = Literal["escalation", "deescalation", "uncertain"]
NewsPhase = Literal[
    "leading_warning",
    "coincident_confirmation",
    "lagging_explanation",
    "phase_uncertain",
]


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    name: str
    date: pd.Timestamp
    category: str
    direction: EventDirection
    description: str
    source_url: str | None = None
    tags: tuple[str, ...] = ()
    current: bool = False
    phase: NewsPhase = "phase_uncertain"
    phase_reason: str = ""
    confirmation_window: str = ""


@dataclass(frozen=True)
class NewsEventClassification:
    category: str
    direction: EventDirection
    confidence: float
    risk_channels: tuple[str, ...]
    candidate_proxies: tuple[str, ...]
    tradable_question: str
    phase: NewsPhase = "phase_uncertain"
    phase_reason: str = ""
    confirmation_window: str = ""


@dataclass(frozen=True)
class EventRiskRun:
    events: tuple[MarketEvent, ...]
    asset_event_returns: pd.DataFrame
    strategy_event_returns: pd.DataFrame
    event_summary: pd.DataFrame
    scenario_playbook: pd.DataFrame
    current_event_scenarios: pd.DataFrame


def load_market_events(path: str | Path | None) -> tuple[MarketEvent, ...]:
    if path is None:
        return ()
    event_path = Path(path)
    if not event_path.exists():
        return ()

    with event_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    events = []
    for item in raw.get("events", []):
        events.append(_market_event_from_mapping(item))
    return tuple(sorted(events, key=lambda event: event.date))


def run_event_risk_study(
    prices: pd.DataFrame,
    results: dict[str, BacktestResult],
    events: tuple[MarketEvent, ...],
    *,
    windows: tuple[int, ...] = DEFAULT_EVENT_WINDOWS,
    asset_proxies: tuple[str, ...] = DEFAULT_EVENT_ASSET_PROXIES,
    primary_strategy: str = "drawdown_managed_dual_momentum",
) -> EventRiskRun:
    clean_prices = prices.sort_index().ffill()
    available_proxies = tuple(ticker for ticker in asset_proxies if ticker in clean_prices.columns)
    asset_event_returns = _asset_event_returns(clean_prices, events, windows, available_proxies)
    strategy_event_returns = _strategy_event_returns(results, events, windows)
    event_summary = summarize_event_windows(
        asset_event_returns,
        strategy_event_returns,
        events,
        primary_strategy=primary_strategy,
    )
    scenario_playbook = build_scenario_playbook(events)
    if scenario_playbook.empty:
        current_event_scenarios = pd.DataFrame()
    else:
        current_event_scenarios = scenario_playbook[scenario_playbook["current_event"]].reset_index(
            drop=True
        )

    return EventRiskRun(
        events=events,
        asset_event_returns=asset_event_returns,
        strategy_event_returns=strategy_event_returns,
        event_summary=event_summary,
        scenario_playbook=scenario_playbook,
        current_event_scenarios=current_event_scenarios,
    )


def classify_news_text(text: str) -> NewsEventClassification:
    normalized = text.lower()
    phase, phase_reason, confirmation_window = _phase_from_text(normalized)

    if _contains_any(
        normalized,
        (
            "openai",
            "anthropic",
            "ai lab",
            "ai labs",
            "inference",
            "training cost",
            "compute cost",
            "gpu cluster",
            "ai capex",
            "capex",
        ),
    ) and _contains_any(
        normalized,
        (
            "loss",
            "losses",
            "burn",
            "negative margin",
            "spending",
            "costs",
            "financials",
            "audited",
            "leaked",
            "revenue",
            "cash flow",
        ),
    ):
        return NewsEventClassification(
            category="ai_unit_economics",
            direction="escalation",
            confidence=0.85,
            risk_channels=(
                "ai_capex",
                "hyperscaler_margins",
                "semiconductor_demand",
                "cloud_credit_quality",
                "market_concentration",
            ),
            candidate_proxies=(
                "QQQ",
                "XLK",
                "SMH",
                "SOXX",
                "IGV",
                "NVDA",
                "MSFT",
                "AVGO",
                "ORCL",
                "PLTR",
                "META",
                "AMZN",
                "HYG",
                "VIXY",
            ),
            tradable_question=(
                "Is AI demand still funding profitable capex, or are AI losses starting to "
                "pressure hyperscaler margins, semis leadership, and market concentration?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_any(
        normalized,
        (
            "data center",
            "datacenter",
            "gpu",
            "nvidia",
            "coreweave",
            "ai infrastructure",
            "ai power",
            "ai electricity",
            "mlperf",
            "semiconductor",
            "chip",
        ),
    ) and _contains_any(
        normalized,
        (
            "ai",
            "artificial intelligence",
            "training",
            "inference",
            "accelerator",
            "power",
            "electricity",
            "capacity",
            "cluster",
        ),
    ):
        ai_infrastructure_direction: EventDirection = "uncertain"
        if _contains_any(
            normalized,
            ("bottleneck", "shortage", "constraint", "strain", "cost", "surge", "delay"),
        ):
            ai_infrastructure_direction = "escalation"
        if _contains_any(normalized, ("expansion", "supply deal", "capacity added", "efficiency")):
            ai_infrastructure_direction = "deescalation"
        return NewsEventClassification(
            category="ai_infrastructure",
            direction=ai_infrastructure_direction,
            confidence=0.68,
            risk_channels=(
                "ai_capex",
                "semiconductor_demand",
                "power_demand",
                "grid_constraint",
                "data_center_buildout",
            ),
            candidate_proxies=(
                "QQQ",
                "XLK",
                "SMH",
                "SOXX",
                "NVDA",
                "VRT",
                "ETN",
                "PWR",
                "CEG",
                "XLU",
                "VIXY",
            ),
            tradable_question=(
                "Is AI infrastructure buildout still expanding profitably, or are power, "
                "chip supply, and capex constraints becoming a market risk?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_any(
        normalized,
        (
            "private credit",
            "direct lending",
            "bdc",
            "business development company",
            "middle market loan",
            "private debt",
            "covenant-lite",
            "covenant lite",
        ),
    ) or _contains_clo_signal(normalized):
        private_credit_direction: EventDirection = "escalation"
        if _contains_any(normalized, ("improve", "recover", "spread tightening", "inflows")):
            private_credit_direction = "deescalation"
        return NewsEventClassification(
            category="private_credit",
            direction=private_credit_direction,
            confidence=0.78,
            risk_channels=(
                "credit",
                "liquidity",
                "regional_banks",
                "levered_credit",
                "small_caps",
            ),
            candidate_proxies=("BIZD", "SRLN", "BKLN", "HYG", "JNK", "LQD", "IWM", "KRE", "VIXY"),
            tradable_question=(
                "Is private-credit stress isolated, or is it leaking into liquid credit, "
                "small caps, and financial conditions?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_monetary_policy_signal(normalized):
        monetary_direction: EventDirection = "uncertain"
        if _contains_any(
            normalized,
            (
                "hawkish",
                "rate hike",
                "hike rates",
                "higher for longer",
                "tightening",
                "quantitative tightening",
                "balance sheet runoff",
                "inflation still elevated",
            ),
        ):
            monetary_direction = "escalation"
        if _contains_any(
            normalized,
            (
                "dovish",
                "rate cut",
                "cut rates",
                "pause",
                "easing",
                "quantitative easing",
                "liquidity facility",
                "repo facility",
            ),
        ):
            monetary_direction = "deescalation"
        return NewsEventClassification(
            category="monetary_policy",
            direction=monetary_direction,
            confidence=0.74,
            risk_channels=("rates", "liquidity", "dollar", "duration", "risk_appetite"),
            candidate_proxies=("TLT", "IEF", "UUP", "GLD", "SPY", "QQQ", "HYG", "VIXY"),
            tradable_question=(
                "Is policy easing enough to support risk assets, or are rates and liquidity "
                "conditions tightening into equity and credit exposure?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_macro_release_signal(normalized):
        macro_direction: EventDirection = "uncertain"
        if _contains_any(
            normalized,
            (
                "hotter than expected",
                "above expectations",
                "sticky inflation",
                "accelerated",
                "jobless claims rose",
                "payrolls missed",
                "recession",
                "contraction",
                "weak demand",
            ),
        ):
            macro_direction = "escalation"
        if macro_direction != "escalation" and _contains_any(
            normalized,
            (
                "cooler than expected",
                "below expectations",
                "disinflation",
                "soft landing",
                "goldilocks",
                "payrolls beat",
                "claims fell",
                "growth accelerated",
            ),
        ):
            macro_direction = "deescalation"
        return NewsEventClassification(
            category="macro_release",
            direction=macro_direction,
            confidence=0.70,
            risk_channels=("growth", "inflation", "rates", "labor", "risk_appetite"),
            candidate_proxies=("SPY", "QQQ", "IWM", "RSP", "TLT", "TIP", "UUP", "VIXY"),
            tradable_question=(
                "Does the macro release change the growth/inflation mix enough to alter "
                "equity beta, duration exposure, or defensive allocation?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_market_plumbing_signal(normalized):
        plumbing_direction: EventDirection = "uncertain"
        if _contains_any(
            normalized,
            (
                "spike",
                "surge",
                "stress",
                "illiquidity",
                "failed auction",
                "funding pressure",
                "margin call",
                "forced selling",
                "deleveraging",
            ),
        ):
            plumbing_direction = "escalation"
        if plumbing_direction != "escalation" and _contains_any(
            normalized,
            ("normalizes", "stabilizes", "eases", "volatility falls", "liquidity improves"),
        ):
            plumbing_direction = "deescalation"
        return NewsEventClassification(
            category="market_plumbing",
            direction=plumbing_direction,
            confidence=0.76,
            risk_channels=("volatility", "funding", "liquidity", "dealer_positioning", "credit"),
            candidate_proxies=("VIXY", "SVXY", "TLT", "UUP", "HYG", "LQD", "SPY", "QQQ"),
            tradable_question=(
                "Is this a contained volatility/liquidity event, or a market-plumbing problem "
                "that should shrink risk before prices fully confirm?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_regulatory_filing_signal(normalized):
        regulatory_direction: EventDirection = "escalation"
        if _contains_any(normalized, ("settlement", "resolved", "dismissed", "approval")):
            regulatory_direction = "deescalation"
        return NewsEventClassification(
            category="regulatory_filing",
            direction=regulatory_direction,
            confidence=0.66,
            risk_channels=(
                "regulatory",
                "accounting",
                "governance",
                "idiosyncratic",
                "sector_risk",
            ),
            candidate_proxies=("SPY", "QQQ", "XLK", "XLF", "KRE", "HYG", "VIXY"),
            tradable_question=(
                "Is the filing/regulatory item idiosyncratic, or does it point to broader "
                "sector, accounting, credit, or governance stress?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_earnings_revision_signal(normalized):
        earnings_direction: EventDirection = "uncertain"
        if _contains_any(
            normalized,
            (
                "miss",
                "missed",
                "cut guidance",
                "lowered guidance",
                "profit warning",
                "downgrade",
                "margin pressure",
                "revenue shortfall",
                "estimates cut",
            ),
        ):
            earnings_direction = "escalation"
        if earnings_direction != "escalation" and _contains_any(
            normalized,
            (
                "beat",
                "beats",
                "raised guidance",
                "raises guidance",
                "upgrade",
                "margin expansion",
                "estimates raised",
            ),
        ):
            earnings_direction = "deescalation"
        return NewsEventClassification(
            category="earnings_revision",
            direction=earnings_direction,
            confidence=0.72,
            risk_channels=(
                "earnings",
                "margins",
                "valuation",
                "sector_rotation",
                "market_concentration",
            ),
            candidate_proxies=("SPY", "QQQ", "RSP", "XLK", "XLY", "XLF", "SMH", "IGV", "VIXY"),
            tradable_question=(
                "Are earnings revisions isolated to one company, or broad enough to change "
                "sector leadership, valuation support, or market concentration risk?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_retail_sentiment_signal(normalized):
        retail_direction: EventDirection = "uncertain"
        if _contains_any(normalized, ("squeeze", "mania", "record call buying", "viral")):
            retail_direction = "escalation"
        if retail_direction != "escalation" and _contains_any(
            normalized,
            ("unwinds", "collapses", "call buying fades", "short interest falls"),
        ):
            retail_direction = "deescalation"
        return NewsEventClassification(
            category="retail_sentiment",
            direction=retail_direction,
            confidence=0.58,
            risk_channels=("crowding", "options", "speculation", "liquidity", "risk_appetite"),
            candidate_proxies=("IWM", "ARKK", "QQQ", "SPHB", "TSLA", "IBIT", "VIXY"),
            tradable_question=(
                "Is retail/speculative pressure broadening risk appetite, or creating crowded "
                "upside that can unwind quickly?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_any(normalized, ("hormuz", "strait", "lng chokepoint", "shipping chokepoint")):
        if _contains_any(normalized, ("reopen", "deal", "ceasefire", "waiver", "flow")):
            return NewsEventClassification(
                category="oil_chokepoint",
                direction="deescalation",
                confidence=0.80,
                risk_channels=("oil", "inflation", "credit", "risk_appetite"),
                candidate_proxies=("USO", "XLE", "DBC", "HYG", "LQD", "SPY", "QQQ", "VIXY"),
                tradable_question=(
                    "Is the oil shock fading fast enough for risk assets to hold, or is this "
                    "a fragile relief rally that needs confirmation?"
                ),
                phase=phase,
                phase_reason=phase_reason,
                confirmation_window=confirmation_window,
            )
        return NewsEventClassification(
            category="oil_chokepoint",
            direction="escalation",
            confidence=0.75,
            risk_channels=("oil", "inflation", "volatility", "safe_haven"),
            candidate_proxies=("USO", "XLE", "DBC", "GLD", "TLT", "UUP", "VIXY", "SPY", "QQQ"),
            tradable_question=(
                "Is this an energy scarcity shock that should reduce equity beta until oil, "
                "credit, and volatility confirm stabilization?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_any(
        normalized,
        (
            "oil",
            "crude",
            "gasoline",
            "diesel",
            "opec",
            "eia",
            "inventory",
            "inventories",
            "refinery",
            "oil sands",
            "brent",
        ),
    ):
        energy_direction: EventDirection = "uncertain"
        if _contains_any(
            normalized,
            (
                "falling inventories",
                "inventories falling",
                "inventories still falling",
                "inventories decreased",
                "inventories declined",
                "inventory draw",
                "stock draw",
                "supply disruption",
                "shortage",
                "surge",
                "spike",
                "sanction",
            ),
        ):
            energy_direction = "escalation"
        if _contains_any(
            normalized,
            ("surplus", "supply returns", "inventories build", "price fell", "glut"),
        ):
            energy_direction = "deescalation"
        return NewsEventClassification(
            category="energy_supply",
            direction=energy_direction,
            confidence=0.68,
            risk_channels=("oil", "inflation", "energy_equities", "rates", "risk_appetite"),
            candidate_proxies=(
                "USO",
                "BNO",
                "XLE",
                "XOP",
                "OIH",
                "DBC",
                "TIP",
                "TLT",
                "SPY",
                "VIXY",
            ),
            tradable_question=(
                "Is this energy news inflationary and risk-negative, or does it lower oil "
                "pressure enough to support broader risk appetite?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_any(normalized, ("tariff", "trade war", "duties", "reciprocal")):
        if _contains_any(normalized, ("pause", "delay", "exempt", "deal", "rollback")):
            return NewsEventClassification(
                category="trade_policy",
                direction="deescalation",
                confidence=0.75,
                risk_channels=("growth", "margins", "inflation", "policy_uncertainty"),
                candidate_proxies=("SPY", "QQQ", "IWM", "SMH", "EEM", "UUP", "TLT", "VIXY"),
                tradable_question=(
                    "Is tariff relief broad and durable enough to re-risk, or is it a "
                    "short-covering rally inside a policy-volatility regime?"
                ),
                phase=phase,
                phase_reason=phase_reason,
                confirmation_window=confirmation_window,
            )
        return NewsEventClassification(
            category="trade_policy",
            direction="escalation",
            confidence=0.80,
            risk_channels=("growth", "margins", "inflation", "policy_uncertainty"),
            candidate_proxies=("SPY", "QQQ", "IWM", "SMH", "EEM", "UUP", "TLT", "VIXY"),
            tradable_question=(
                "Is this a growth and margin shock that should cut cyclical/AI beta until "
                "breadth and credit stop deteriorating?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    if _contains_military_escalation_signal(normalized):
        return NewsEventClassification(
            category="military_escalation",
            direction="escalation",
            confidence=0.65,
            risk_channels=("volatility", "safe_haven", "oil", "defense"),
            candidate_proxies=("SPY", "QQQ", "GLD", "TLT", "USO", "XLE", "ITA", "PPA", "VIXY"),
            tradable_question=(
                "Is the shock localized and fading, or does it create a broader volatility "
                "and safe-haven rotation?"
            ),
            phase=phase,
            phase_reason=phase_reason,
            confirmation_window=confirmation_window,
        )

    return NewsEventClassification(
        category="unclassified",
        direction="uncertain",
        confidence=0.25,
        risk_channels=("unknown",),
        candidate_proxies=("SPY", "QQQ", "BIL", "VIXY"),
        tradable_question=(
            "What market channel does this news affect: growth, inflation, liquidity, credit, "
            "concentration, or volatility?"
        ),
        phase=phase,
        phase_reason=phase_reason,
        confirmation_window=confirmation_window,
    )


def summarize_event_windows(
    asset_event_returns: pd.DataFrame,
    strategy_event_returns: pd.DataFrame,
    events: tuple[MarketEvent, ...],
    *,
    primary_strategy: str,
) -> pd.DataFrame:
    if asset_event_returns.empty:
        return pd.DataFrame()

    event_lookup = {event.event_id: event for event in events}
    asset_complete = asset_event_returns[asset_event_returns["complete"]].copy()
    if asset_complete.empty:
        return pd.DataFrame()

    asset_pivot = asset_complete.pivot_table(
        index=["event_id", "window", "horizon_trading_days"],
        columns="ticker",
        values="return",
        aggfunc="first",
    )
    strategy_pivot = pd.DataFrame()
    if not strategy_event_returns.empty:
        strategy_pivot = strategy_event_returns[strategy_event_returns["complete"]].pivot_table(
            index=["event_id", "window", "horizon_trading_days"],
            columns="strategy",
            values="return",
            aggfunc="first",
        )

    rows: list[dict[str, object]] = []
    for key, asset_row in asset_pivot.iterrows():
        event_id, window, horizon = key
        event = event_lookup[str(event_id)]
        strategy_row = (
            strategy_pivot.loc[key]
            if not strategy_pivot.empty and key in strategy_pivot.index
            else pd.Series(dtype=float)
        )
        risk_asset_return = _basket_return(asset_row, ("SPY", "QQQ", "IWM", "SMH", "HYG"))
        defensive_return = _basket_return(asset_row, ("BIL", "TLT", "GLD"))
        oil_return = _basket_return(asset_row, ("USO", "XLE", "DBC"))
        credit_return = _relative_return(asset_row, "HYG", "LQD")
        vixy_return = _safe_get(asset_row, "VIXY")
        best_strategy = _best_strategy(strategy_row)

        rows.append(
            {
                "event_id": event.event_id,
                "event_name": event.name,
                "event_date": str(event.date.date()),
                "category": event.category,
                "direction": event.direction,
                "event_phase": event.phase,
                "phase_reason": event.phase_reason,
                "confirmation_window": event.confirmation_window,
                "window": window,
                "horizon_trading_days": horizon,
                "risk_asset_return": risk_asset_return,
                "defensive_return": defensive_return,
                "oil_complex_return": oil_return,
                "credit_relative_return": credit_return,
                "vixy_return": vixy_return,
                "spy_return": _safe_get(asset_row, "SPY"),
                "qqq_return": _safe_get(asset_row, "QQQ"),
                "primary_strategy_return": _safe_get(strategy_row, primary_strategy),
                "best_strategy": best_strategy[0],
                "best_strategy_return": best_strategy[1],
                "market_mode": _market_mode(
                    risk_asset_return, defensive_return, oil_return, vixy_return
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_date", "horizon_trading_days"])


def build_scenario_playbook(events: tuple[MarketEvent, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events:
        for scenario in _scenario_templates(event.category, event.direction):
            rows.append(
                {
                    "event_id": event.event_id,
                    "event_name": event.name,
                    "event_date": str(event.date.date()),
                    "current_event": event.current,
                    "event_phase": event.phase,
                    "phase_reason": event.phase_reason,
                    "confirmation_window": event.confirmation_window,
                    "category": event.category,
                    "direction": event.direction,
                    **scenario,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["current_event", "event_date"], ascending=[False, False])


def _asset_event_returns(
    prices: pd.DataFrame,
    events: tuple[MarketEvent, ...],
    windows: tuple[int, ...],
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events:
        for ticker in tickers:
            series = prices[ticker].dropna()
            for horizon in windows:
                window_return = _level_window_return(series, event.date, horizon)
                rows.append(
                    {
                        **_event_fields(event),
                        "ticker": ticker,
                        "window": _window_label(horizon),
                        "horizon_trading_days": horizon,
                        **_window_return_fields(window_return),
                    }
                )
    return pd.DataFrame(rows)


def _strategy_event_returns(
    results: dict[str, BacktestResult],
    events: tuple[MarketEvent, ...],
    windows: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events:
        for strategy, result in results.items():
            series = result.equity.dropna()
            for horizon in windows:
                window_return = _level_window_return(series, event.date, horizon)
                rows.append(
                    {
                        **_event_fields(event),
                        "strategy": strategy,
                        "window": _window_label(horizon),
                        "horizon_trading_days": horizon,
                        **_window_return_fields(window_return),
                    }
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class _WindowReturn:
    return_: float
    start_date: str | None
    end_date: str | None
    anchor_date: str | None
    complete: bool


def _level_window_return(
    series: pd.Series, event_date: pd.Timestamp, horizon: int
) -> _WindowReturn:
    if series.empty:
        return _empty_window_return()

    sorted_series = series.sort_index().dropna()
    anchor_position = int(sorted_series.index.searchsorted(event_date, side="left"))
    if anchor_position >= len(sorted_series):
        return _empty_window_return()

    if horizon < 0:
        start_position = anchor_position + horizon
        end_position = anchor_position
    else:
        start_position = anchor_position
        end_position = anchor_position + horizon

    anchor_date = str(sorted_series.index[anchor_position].date())
    if start_position < 0 or end_position >= len(sorted_series):
        return _WindowReturn(
            return_=np.nan,
            start_date=None,
            end_date=None,
            anchor_date=anchor_date,
            complete=False,
        )

    start_value = float(sorted_series.iloc[start_position])
    end_value = float(sorted_series.iloc[end_position])
    if start_value <= 0:
        value = np.nan
        complete = False
    else:
        value = end_value / start_value - 1.0
        complete = True

    return _WindowReturn(
        return_=value,
        start_date=str(sorted_series.index[start_position].date()),
        end_date=str(sorted_series.index[end_position].date()),
        anchor_date=anchor_date,
        complete=complete,
    )


def _empty_window_return() -> _WindowReturn:
    return _WindowReturn(
        return_=np.nan,
        start_date=None,
        end_date=None,
        anchor_date=None,
        complete=False,
    )


def _window_return_fields(window_return: _WindowReturn) -> dict[str, object]:
    return {
        "return": window_return.return_,
        "start_date": window_return.start_date,
        "end_date": window_return.end_date,
        "anchor_date": window_return.anchor_date,
        "complete": window_return.complete,
    }


def _market_event_from_mapping(raw: dict[str, Any]) -> MarketEvent:
    inferred_phase, inferred_phase_reason, inferred_confirmation_window = _phase_from_text(
        " ".join(
            [
                str(raw.get("name", "")),
                str(raw.get("description", "")),
                " ".join(str(tag) for tag in raw.get("tags", [])),
            ]
        ).lower()
    )
    return MarketEvent(
        event_id=str(raw["event_id"]),
        name=str(raw["name"]),
        date=pd.Timestamp(raw["date"]),
        category=str(raw["category"]),
        direction=_direction(raw["direction"]),
        description=str(raw.get("description", "")),
        source_url=raw.get("source_url"),
        tags=tuple(str(tag) for tag in raw.get("tags", [])),
        current=bool(raw.get("current", False)),
        phase=_phase(raw.get("phase", inferred_phase)),
        phase_reason=str(raw.get("phase_reason", inferred_phase_reason)),
        confirmation_window=str(raw.get("confirmation_window", inferred_confirmation_window)),
    )


def _direction(value: Any) -> EventDirection:
    direction = str(value)
    if direction not in {"escalation", "deescalation", "uncertain"}:
        msg = f"Unsupported event direction: {direction}"
        raise ValueError(msg)
    return cast(EventDirection, direction)


def _phase(value: Any) -> NewsPhase:
    phase = str(value)
    if phase not in {
        "leading_warning",
        "coincident_confirmation",
        "lagging_explanation",
        "phase_uncertain",
    }:
        msg = f"Unsupported news phase: {phase}"
        raise ValueError(msg)
    return cast(NewsPhase, phase)


def _event_fields(event: MarketEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "event_name": event.name,
        "event_date": str(event.date.date()),
        "category": event.category,
        "direction": event.direction,
        "current_event": event.current,
        "event_phase": event.phase,
        "phase_reason": event.phase_reason,
        "confirmation_window": event.confirmation_window,
    }


def _window_label(horizon: int) -> str:
    if horizon < 0:
        return f"pre_{abs(horizon)}d"
    return f"post_{horizon}d"


def _basket_return(row: pd.Series, tickers: tuple[str, ...]) -> float:
    values = [_safe_get(row, ticker) for ticker in tickers]
    clean = [value for value in values if pd.notna(value)]
    if not clean:
        return np.nan
    return float(np.mean(clean))


def _relative_return(row: pd.Series, numerator: str, denominator: str) -> float:
    numerator_return = _safe_get(row, numerator)
    denominator_return = _safe_get(row, denominator)
    if pd.isna(numerator_return) or pd.isna(denominator_return):
        return np.nan
    return float(numerator_return - denominator_return)


def _safe_get(row: pd.Series, key: str) -> float:
    if key not in row.index or pd.isna(row[key]):
        return np.nan
    return float(row[key])


def _best_strategy(row: pd.Series) -> tuple[str | None, float]:
    clean = row.dropna()
    if clean.empty:
        return None, np.nan
    name = str(clean.idxmax())
    return name, float(clean.loc[name])


def _market_mode(
    risk_asset_return: float,
    defensive_return: float,
    oil_return: float,
    vixy_return: float,
) -> str:
    if pd.notna(oil_return) and oil_return > 0.05 and risk_asset_return < 0:
        return "inflation/risk-off shock"
    if (
        pd.notna(risk_asset_return)
        and risk_asset_return > 0
        and pd.notna(vixy_return)
        and vixy_return < 0
    ):
        return "risk-on relief"
    if pd.notna(risk_asset_return) and risk_asset_return < -0.02 and defensive_return > 0:
        return "safe-haven risk-off"
    if pd.notna(risk_asset_return) and risk_asset_return > 0.02 and oil_return < 0:
        return "disinflationary risk-on"
    return "mixed/transition"


def _scenario_templates(category: str, direction: EventDirection) -> tuple[dict[str, str], ...]:
    if category == "oil_chokepoint" or category == "oil_geopolitical":
        return (
            {
                "scenario": "Energy relief holds",
                "confirmation": "USO/DBC fade, HYG/LQD improves, VIXY falls, SPY/QQQ hold trend.",
                "risk_posture": "Maintain existing risk exposure; add only after two or more sessions confirm.",
                "off_ramp": "Cut risk if oil retraces more than half the relief move while credit weakens.",
            },
            {
                "scenario": "Deal or ceasefire fails",
                "confirmation": "USO, UUP, and VIXY rise together while HYG/LQD and breadth weaken.",
                "risk_posture": "Reduce equity beta, prefer BIL/T-bill posture, and wait for credit stabilization.",
                "off_ramp": "Exit marginal growth/AI exposure if QQQ/RSP and SMH/SPY both roll over.",
            },
            {
                "scenario": "Stagflation pressure",
                "confirmation": "Oil and commodities rise, long-duration bonds fall, equities lose breadth.",
                "risk_posture": "Avoid adding duration-sensitive growth; monitor XLE/GLD as diversifiers.",
                "off_ramp": "Keep cash/T-bill allocation high until oil volatility and inflation proxies settle.",
            },
        )

    if category == "trade_policy":
        return (
            {
                "scenario": "Policy shock broadens",
                "confirmation": "Small caps, semis, and EEM underperform while UUP/VIXY rise.",
                "risk_posture": "Reduce cyclical and AI-beta allocations until breadth and credit recover.",
                "off_ramp": "Do not buy the dip if SPY, QQQ, and HYG are all below trend.",
            },
            {
                "scenario": "Negotiation relief rally",
                "confirmation": "SPY/QQQ regain trend, HYG/LQD improves, VIXY falls for several sessions.",
                "risk_posture": "Re-risk gradually; treat first-day spikes as unconfirmed relief.",
                "off_ramp": "Reverse if relief is isolated to mega-cap growth while equal-weight breadth fails.",
            },
            {
                "scenario": "Policy whipsaw",
                "confirmation": "Large one-day reversals and repeated tariff headlines without breadth follow-through.",
                "risk_posture": "Lower position size and require stronger persistence filters before rotating.",
                "off_ramp": "Prefer BIL/cash when trend signals flip faster than the minimum hold window.",
            },
        )

    if category == "ai_unit_economics":
        return (
            {
                "scenario": "AI profitability concern is ignored",
                "confirmation": "QQQ/SMH/SOXX keep leadership and hyperscalers hold margins narrative.",
                "risk_posture": "Do not fight confirmed AI leadership, but avoid increasing concentration size.",
                "off_ramp": "Reduce AI beta if QQQ/RSP and SMH/SPY both roll over after the news.",
            },
            {
                "scenario": "AI capex repricing starts",
                "confirmation": "SMH/SOXX underperform, hyperscalers weaken, credit/volatility stops confirming risk-on.",
                "risk_posture": "Cut AI satellite risk, cap QQQ exposure, and wait for breadth repair.",
                "off_ramp": "Exit marginal AI exposure if semis fail while HYG/LQD weakens.",
            },
            {
                "scenario": "Cloud margin rotation",
                "confirmation": "MSFT/AMZN/GOOGL weaken versus SPY while non-AI sectors broaden.",
                "risk_posture": "Favor broad market or factor rotation over concentrated mega-cap AI.",
                "off_ramp": "Do not re-add mega-cap AI until cloud margin concern fades in price action.",
            },
        )

    if category == "ai_infrastructure":
        return (
            {
                "scenario": "AI buildout remains constructive",
                "confirmation": "SMH/SOXX and power-infrastructure names hold leadership without credit stress.",
                "risk_posture": "Keep AI exposure sized by concentration controls; prefer confirmed leaders.",
                "off_ramp": "Avoid adding if AI infrastructure rallies while QQQ/RSP breadth deteriorates.",
            },
            {
                "scenario": "Power or capex constraint emerges",
                "confirmation": "VRT/ETN/PWR or semis roll over while hyperscalers underperform.",
                "risk_posture": "Reduce AI satellite risk and wait for infrastructure bottleneck clarity.",
                "off_ramp": "Cut marginal AI exposure if SMH/SPY and QQQ/RSP both break lower.",
            },
            {
                "scenario": "Infrastructure rotation broadens",
                "confirmation": "Power, grid, and industrial names outperform while mega-cap AI concentration cools.",
                "risk_posture": "Favor diversified broad-market or infrastructure proxies over single-name AI beta.",
                "off_ramp": "Reverse if the rotation becomes defensive and credit weakens.",
            },
        )

    if category == "private_credit":
        return (
            {
                "scenario": "Private-credit stress stays contained",
                "confirmation": "BIZD/SRLN/BKLN stabilize and HYG/LQD avoids deterioration.",
                "risk_posture": "Keep core posture but monitor credit and small-cap exposure closely.",
                "off_ramp": "Reduce risk if liquid credit starts confirming the private-market stress.",
            },
            {
                "scenario": "Stress leaks into liquid credit",
                "confirmation": "HYG/LQD weakens, BKLN/SRLN fall, IWM and regional banks lag.",
                "risk_posture": "Cut small-cap/high-beta exposure and favor BIL/T-bills.",
                "off_ramp": "Stay defensive until credit ETFs stop making lower lows.",
            },
            {
                "scenario": "Liquidity-driven risk-off",
                "confirmation": "Credit weakens together with VIXY/UUP strength and poor breadth.",
                "risk_posture": "Prioritize capital preservation; do not average into credit stress.",
                "off_ramp": "Require credit, breadth, and volatility confirmation before re-risking.",
            },
        )

    if category == "energy_supply":
        return (
            {
                "scenario": "Energy pressure stays contained",
                "confirmation": "USO/BNO stop rising, inflation breakevens stabilize, SPY/QQQ hold trend.",
                "risk_posture": "Keep core posture; avoid treating every oil headline as a de-risk trigger.",
                "off_ramp": "Reduce risk if oil strength starts lifting volatility and hurting breadth.",
            },
            {
                "scenario": "Inflationary oil shock",
                "confirmation": "USO/BNO/XLE rise while TLT falls, VIXY rises, and growth stocks weaken.",
                "risk_posture": "Lower duration-sensitive growth exposure and increase defensive allocation.",
                "off_ramp": "Stay cautious until oil momentum and rates pressure fade together.",
            },
            {
                "scenario": "Oil surplus or relief",
                "confirmation": "Oil proxies fall, inflation pressure eases, and breadth/credit improve.",
                "risk_posture": "Allow risk-on signals to reassert if trend and credit confirm.",
                "off_ramp": "Do not chase relief if lower oil is paired with demand/growth deterioration.",
            },
        )

    if category == "monetary_policy":
        return (
            {
                "scenario": "Policy relief broadens",
                "confirmation": "TLT stabilizes, UUP fades, HYG/LQD improves, and SPY/QQQ breadth expands.",
                "risk_posture": "Allow risk budget to rise gradually if price confirmation follows the policy signal.",
                "off_ramp": "Reduce risk if rates or the dollar rise despite dovish interpretation.",
            },
            {
                "scenario": "Hawkish rates shock",
                "confirmation": "TLT falls, UUP rises, QQQ/RSP weakens, and credit spreads pressure risk assets.",
                "risk_posture": "Cap duration-sensitive growth and prefer defensive allocation until rates stabilize.",
                "off_ramp": "Do not re-risk until TLT, credit, and breadth stop deteriorating together.",
            },
            {
                "scenario": "Liquidity ambiguity",
                "confirmation": "Mixed Fed language with unstable reactions across UUP, HYG, TLT, and QQQ.",
                "risk_posture": "Keep position sizes smaller and wait for cross-asset confirmation.",
                "off_ramp": "Treat failed relief rallies as a signal that liquidity is tighter than headlines imply.",
            },
        )

    if category == "macro_release":
        return (
            {
                "scenario": "Goldilocks confirmation",
                "confirmation": "Rates ease, breadth improves, and cyclical/risk assets hold trend after the release.",
                "risk_posture": "Permit risk-on posture if trend, breadth, and credit confirm.",
                "off_ramp": "Reverse if lower rates are paired with growth-scare leadership and weak credit.",
            },
            {
                "scenario": "Hot inflation or rates repricing",
                "confirmation": "TLT falls, UUP rises, inflation proxies firm, and QQQ/SMH lose leadership.",
                "risk_posture": "Reduce duration-sensitive growth and avoid adding high-beta exposure.",
                "off_ramp": "Stay smaller until rates pressure fades and breadth repairs.",
            },
            {
                "scenario": "Growth scare",
                "confirmation": "IWM/RSP/cyclicals weaken, credit lags, and defensive assets lead.",
                "risk_posture": "Prefer defensive allocation while waiting for credit and breadth stabilization.",
                "off_ramp": "Do not treat lower yields as bullish if credit and small caps deteriorate.",
            },
        )

    if category == "earnings_revision":
        return (
            {
                "scenario": "Earnings support persists",
                "confirmation": "Guidance strength broadens beyond mega-cap leaders and RSP improves versus SPY.",
                "risk_posture": "Maintain risk exposure if revisions confirm breadth rather than concentration.",
                "off_ramp": "Avoid adding if earnings strength is narrow and credit/volatility deteriorate.",
            },
            {
                "scenario": "Guidance cuts broaden",
                "confirmation": "Multiple sectors cut guidance while margins, credit, and breadth weaken together.",
                "risk_posture": "Cut cyclical and high-beta exposure; favor defensive allocation.",
                "off_ramp": "Wait for estimate stabilization before rebuilding exposure.",
            },
            {
                "scenario": "Leadership rotation",
                "confirmation": "Earnings winners shift away from crowded AI/mega-cap into broader sectors.",
                "risk_posture": "Favor diversified broad-market or factor rotation over concentrated growth.",
                "off_ramp": "Reverse if rotation becomes defensive rather than broadening.",
            },
        )

    if category == "market_plumbing":
        return (
            {
                "scenario": "Volatility spike fades",
                "confirmation": "VIXY falls, HYG/LQD repairs, UUP stabilizes, and equities regain trend.",
                "risk_posture": "Re-risk gradually only after liquidity stress visibly fades.",
                "off_ramp": "Cut risk again if volatility rebounds while credit weakens.",
            },
            {
                "scenario": "Liquidity accident",
                "confirmation": "VIXY/UUP rise together, credit sells off, and risk assets gap through trend.",
                "risk_posture": "Prioritize drawdown control and defensive allocation over dip buying.",
                "off_ramp": "Require volatility, credit, and breadth confirmation before re-risking.",
            },
            {
                "scenario": "Positioning squeeze",
                "confirmation": "Sharp reversal led by crowded assets without broad credit or breadth support.",
                "risk_posture": "Avoid chasing one-day squeezes; require persistence across several sessions.",
                "off_ramp": "Exit marginal exposure if the squeeze fails and VIXY/UUP rise.",
            },
        )

    if category == "regulatory_filing":
        return (
            {
                "scenario": "Idiosyncratic filing risk",
                "confirmation": "Affected names weaken but sector ETFs, credit, and broad indexes remain stable.",
                "risk_posture": "Do not change portfolio risk unless sector or credit confirmation appears.",
                "off_ramp": "Escalate if related peers or credit proxies start confirming contagion.",
            },
            {
                "scenario": "Sector overhang",
                "confirmation": "Regulatory or accounting news spreads across sector ETFs and peers.",
                "risk_posture": "Reduce affected sector/theme exposure until peer confirmation fades.",
                "off_ramp": "Rebuild only after sector relative strength and credit stabilize.",
            },
            {
                "scenario": "Governance or accounting contagion",
                "confirmation": "Restatement/material weakness headlines pressure credit and high-beta equities.",
                "risk_posture": "Treat as left-tail risk and shrink concentrated exposure.",
                "off_ramp": "Stay smaller until volatility and sector breadth normalize.",
            },
        )

    if category == "retail_sentiment":
        return (
            {
                "scenario": "Speculative risk-on broadens",
                "confirmation": "Retail/call activity coincides with broad market breadth and credit improvement.",
                "risk_posture": "Permit risk-on posture only if speculation is supported by broader confirmation.",
                "off_ramp": "Avoid adding if speculative assets run while breadth and credit diverge.",
            },
            {
                "scenario": "Crowded squeeze unwind",
                "confirmation": "Meme/high-beta names reverse lower while VIXY rises and liquidity fades.",
                "risk_posture": "Reduce high-beta satellite exposure and avoid chasing late-stage squeezes.",
                "off_ramp": "Wait for volatility and breadth repair before re-entering speculative themes.",
            },
            {
                "scenario": "Noise without confirmation",
                "confirmation": "Social chatter is elevated but major proxies do not confirm the move.",
                "risk_posture": "Keep as triage context; do not override numerical risk engine.",
                "off_ramp": "Ignore unless it begins affecting trend, volatility, or credit.",
            },
        )

    if category == "military_escalation":
        return (
            {
                "scenario": "Contained strike",
                "confirmation": "Gold/oil spike fades and equities recover within one to five sessions.",
                "risk_posture": "Avoid panic exits if trend and credit remain intact.",
                "off_ramp": "Reduce risk if the shock moves from localized strike to supply-chain disruption.",
            },
            {
                "scenario": "Escalation spiral",
                "confirmation": "VIXY and safe havens rise while SPY/QQQ/credit break trend.",
                "risk_posture": "Cut equity beta and prioritize drawdown control over forecast conviction.",
                "off_ramp": "Stay defensive until volatility falls and risk assets reclaim trend.",
            },
        )

    return (
        {
            "scenario": f"{direction.title()} event needs classification",
            "confirmation": "Map the news to growth, inflation, liquidity, credit, or volatility channels.",
            "risk_posture": "Do not override numerical signals until the affected market channel is clear.",
            "off_ramp": "Stay defensive if volatility rises and breadth/credit deteriorate together.",
        },
    )


def _contains_monetary_policy_signal(text: str) -> bool:
    return _contains_any(
        text,
        (
            "federal reserve",
            "fomc",
            "powell",
            "fed ",
            "central bank",
            "rate cut",
            "rate hike",
            "policy rate",
            "dot plot",
            "quantitative tightening",
            "quantitative easing",
            "balance sheet runoff",
            "reverse repo",
            "bank reserves",
            "discount window",
        ),
    )


def _contains_macro_release_signal(text: str) -> bool:
    return _contains_any(
        text,
        (
            "cpi",
            "core pce",
            "pce inflation",
            "ppi",
            "nonfarm payroll",
            "payrolls",
            "jobs report",
            "jobless claims",
            "unemployment rate",
            "retail sales",
            "industrial production",
            "consumer sentiment",
            "ism",
            "pmi",
            "gdp",
            "durable goods",
        ),
    )


def _contains_earnings_revision_signal(text: str) -> bool:
    return _contains_any(
        text,
        (
            "earnings",
            "guidance",
            "eps",
            "revenue",
            "margin",
            "profit warning",
            "analyst downgrade",
            "analyst upgrade",
            "estimate cut",
            "estimates cut",
            "estimate raised",
            "estimates raised",
            "lowered forecast",
            "raised forecast",
            "cut outlook",
            "raises outlook",
            "revision",
            "revisions",
        ),
    )


def _contains_market_plumbing_signal(text: str) -> bool:
    return _contains_any(
        text,
        (
            "vix",
            "move index",
            "skew",
            "put/call",
            "put call",
            "dealer gamma",
            "0dte",
            "options volume",
            "treasury auction",
            "repo market",
            "funding market",
            "basis trade",
            "market liquidity",
            "dollar funding",
            "yen carry",
            "forced selling",
            "margin call",
        ),
    )


def _contains_regulatory_filing_signal(text: str) -> bool:
    has_regulatory_context = _contains_any(
        text,
        (
            "sec",
            "ftc",
            "doj",
            "antitrust",
            "subpoena",
            "wells notice",
            "8-k",
            "10-k",
            "10-q",
            "auditor",
        ),
    )
    has_market_risk_context = _contains_any(
        text,
        (
            "investigation",
            "probe",
            "enforcement",
            "fraud",
            "restatement",
            "material weakness",
            "accounting",
            "going concern",
            "bankruptcy",
            "delisting",
            "settlement",
            "approval",
        ),
    )
    return has_regulatory_context and has_market_risk_context


def _contains_retail_sentiment_signal(text: str) -> bool:
    return _contains_any(
        text,
        (
            "reddit",
            "wallstreetbets",
            "meme stock",
            "meme stocks",
            "short squeeze",
            "gamma squeeze",
            "retail traders",
            "call buying",
            "social media traders",
            "roaring kitty",
            "most shorted",
        ),
    )


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _contains_clo_signal(text: str) -> bool:
    return bool(re.search(r"\bclos?\b|\bcollateralized loan obligations?\b", text))


def _contains_military_escalation_signal(text: str) -> bool:
    if _contains_any(
        text,
        (
            "missile",
            "bomb",
            "retaliat",
            "airstrike",
            "air strike",
            "drone strike",
            "military strike",
        ),
    ):
        return True
    return bool(re.search(r"\bwars?\b", text))


def _phase_from_text(text: str) -> tuple[NewsPhase, str, str]:
    if _contains_any(
        text,
        (
            "exclusive",
            "leaked",
            "audited",
            "documents",
            "revealed",
            "first reported",
            "new data",
        ),
    ):
        return (
            "leading_warning",
            "Document/news disclosure may precede full market repricing.",
            "Watch 1d/5d/21d confirmation in affected proxies.",
        )
    if _contains_any(
        text,
        (
            "shares fell",
            "stocks fell",
            "market reacted",
            "selloff",
            "rally",
            "after the report",
            "following the report",
        ),
    ):
        return (
            "coincident_confirmation",
            "News appears to be arriving with an active price response.",
            "Check whether 1d move persists over 5d and broadens across proxies.",
        )
    if _contains_any(
        text,
        (
            "explains",
            "attributed to",
            "because of",
            "analysts said after",
            "retrospective",
            "postmortem",
        ),
    ):
        return (
            "lagging_explanation",
            "News may be explaining price action already underway.",
            "Avoid reacting unless trend, breadth, or credit confirms a fresh leg.",
        )
    return (
        "phase_uncertain",
        "No clear leading/coincident/lagging phase marker in text.",
        "Require market confirmation before changing posture.",
    )
