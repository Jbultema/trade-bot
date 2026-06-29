from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_NARRATIVE_AI_INFRASTRUCTURE_TICKERS,
    DEFAULT_NARRATIVE_AI_SUPPLIER_TICKERS,
    DEFAULT_NARRATIVE_DEFENSIVE_CONFIRMATION_TICKERS,
    DEFAULT_NARRATIVE_GLOBAL_CHIP_TICKERS,
    DEFAULT_NARRATIVE_HYPERSCALER_TICKERS,
    DEFAULT_NARRATIVE_OPERATING_DATA_SUPPORT,
    DEFAULT_NARRATIVE_RESEARCH_ONLY_DATA_SUPPORT,
    DEFAULT_NARRATIVE_SIGNAL_ABSORPTION_LOOKBACK_DAYS,
    DEFAULT_NARRATIVE_SIGNAL_ACTIVE_SCORE,
    DEFAULT_NARRATIVE_SIGNAL_DECISION_ROLE,
    DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS,
    DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS,
    DEFAULT_NARRATIVE_SIGNAL_MODEL_AUTHORITY,
    DEFAULT_NARRATIVE_SIGNAL_NEWS_URGENCY_THRESHOLD,
    DEFAULT_NARRATIVE_SIGNAL_PROMOTION_REQUIREMENT,
    DEFAULT_NARRATIVE_SIGNAL_RELATIVE_STRENGTH_THRESHOLD,
    DEFAULT_NARRATIVE_SIGNAL_STRONG_RELATIVE_STRENGTH,
    DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE,
    DEFAULT_NARRATIVE_SPECULATIVE_TICKERS,
    DEFAULT_NARRATIVE_UNSUPPORTED_DATA_SUPPORT,
)
from trade_bot.research.event_risk import MarketEvent

NARRATIVE_SIGNAL_COLUMNS = [
    "signal_id",
    "signal_name",
    "source_threads",
    "data_support",
    "score",
    "status",
    "direction",
    "evidence",
    "read_through",
    "decision_role",
    "model_authority",
    "promotion_requirement",
    "data_used",
    "missing_data",
    "trade_use",
]


@dataclass(frozen=True)
class _Signal:
    signal_id: str
    signal_name: str
    source_threads: str
    data_support: str
    score: float
    direction: str
    evidence: str
    read_through: str
    data_used: str
    missing_data: str
    decision_role: str = DEFAULT_NARRATIVE_SIGNAL_DECISION_ROLE
    model_authority: str = DEFAULT_NARRATIVE_SIGNAL_MODEL_AUTHORITY
    promotion_requirement: str = DEFAULT_NARRATIVE_SIGNAL_PROMOTION_REQUIREMENT
    trade_use: str = "Research context only until experiment evidence says it improves outcomes."


def build_narrative_signal_table(
    prices: pd.DataFrame,
    *,
    news_triage: pd.DataFrame | None = None,
    events: tuple[MarketEvent, ...] = (),
) -> pd.DataFrame:
    """Score cross-source investor themes using only data the project has.

    The table intentionally separates proxy-backed signals from unsupported
    watchlist gaps. It should improve interpretation and experiment design, not
    act as a hidden discretionary trading override.
    """

    clean_prices = prices.sort_index().ffill()
    triage = news_triage if news_triage is not None else pd.DataFrame()
    signals = [
        _ai_supplier_rotation(clean_prices, triage, events),
        _hyperscaler_capex_pressure(clean_prices, triage, events),
        _ai_inflation_pass_through(clean_prices, triage, events),
        _concentration_broadening(clean_prices),
        _oil_inflation_shock(clean_prices, triage, events),
        _private_credit_liquidity(clean_prices, triage, events),
        _policy_put_uncertainty(clean_prices, triage, events),
        _speculative_leverage_proxy(clean_prices, triage, events),
        _ipo_equity_supply_pressure(triage, events),
        _positive_catalyst_absorption(clean_prices, triage, events),
        _international_chip_concentration(clean_prices),
        _sector_valuation_policy_proxy(clean_prices, triage, events),
        _easy_bubble_vs_hard_risk_off(clean_prices, triage, events),
        _unsupported_data_watchlist(),
    ]
    frame = pd.DataFrame(
        [{**signal.__dict__, "status": _status(signal.score, signal.data_support)} for signal in signals],
        columns=NARRATIVE_SIGNAL_COLUMNS,
    )
    return frame.sort_values(
        ["status", "score"],
        key=lambda series: series.map(_status_rank) if series.name == "status" else series,
        ascending=[True, False],
    ).reset_index(drop=True)


def summarize_narrative_signals(signals: pd.DataFrame) -> dict[str, str]:
    if signals.empty:
        return {
            "answer": "No narrative signal table",
            "detail": (
                "Run the current-state pipeline to build cross-source insight diagnostics. "
                "These diagnostics have no direct allocation authority."
            ),
            "plain": "no cross-source narrative diagnostics are loaded",
            "tone": "neutral",
        }
    supported = signals[signals["data_support"].isin(DEFAULT_NARRATIVE_OPERATING_DATA_SUPPORT)]
    active = supported[supported["status"].isin(["active", "warning"])]
    research_only = signals[
        signals["data_support"].isin(DEFAULT_NARRATIVE_RESEARCH_ONLY_DATA_SUPPORT)
    ]
    unsupported = signals[
        signals["data_support"].isin(DEFAULT_NARRATIVE_UNSUPPORTED_DATA_SUPPORT)
    ]
    if active.empty:
        return {
            "answer": "No active cross-source pressure",
            "detail": (
                "Supported external-source themes are visible, but none are scoring as active pressure. "
                "This layer is explainer/research-only until promoted by ablation tests."
            ),
            "plain": "no supported cross-source narrative signal is active",
            "tone": "neutral",
        }
    top = active.sort_values("score", ascending=False).head(3)
    themes = ", ".join(str(value) for value in top["signal_name"])
    unsupported_count = int(len(unsupported))
    research_only_count = int(len(research_only))
    return {
        "answer": f"{len(active)} active narrative signal(s)",
        "detail": (
            f"Top active operating-context themes: {themes}. Research-only thin proxies: "
            f"{research_only_count}; unsupported watchlist gaps: {unsupported_count}. Decision role: "
            "explainer/research-only, not a direct sizing input."
        ),
        "plain": f"{len(active)} active narrative signal(s): {themes}",
        "tone": "warning" if float(top["score"].max()) < DEFAULT_NARRATIVE_SIGNAL_ACTIVE_SCORE else "critical",
    }


def _ai_supplier_rotation(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    supplier = _basket_relative_return(
        prices, DEFAULT_NARRATIVE_AI_SUPPLIER_TICKERS, "SPY", DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS
    )
    hyperscaler = _basket_relative_return(
        prices, DEFAULT_NARRATIVE_HYPERSCALER_TICKERS, "SPY", DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS
    )
    spread = supplier - hyperscaler
    news = _news_pressure(triage, events, categories=("ai_infrastructure", "ai_unit_economics"))
    score = _clip01(0.55 * _scale_strength(spread, DEFAULT_NARRATIVE_SIGNAL_STRONG_RELATIVE_STRENGTH) + 0.45 * news)
    direction = "AI suppliers leading hyperscalers" if spread > 0 else "No confirmed supplier leadership"
    return _Signal(
        signal_id="ai_supplier_hyperscaler_divergence",
        signal_name="AI supplier / hyperscaler divergence",
        source_threads="AI capex, platform economics, weekly market wrap, and macro-risk process commentary",
        data_support="proxy",
        score=score,
        direction=direction,
        evidence=f"Supplier basket vs SPY {supplier:.1%}; hyperscaler basket vs SPY {hyperscaler:.1%}; spread {spread:.1%}.",
        read_through="Checks whether the AI trade is rotating from platform owners toward silicon, memory, power, and infrastructure suppliers.",
        data_used="Yahoo price proxies plus AI infrastructure/unit-economics news categories.",
        missing_data="Company segment-level capex commitments, contract margins, and order-book data.",
    )


def _hyperscaler_capex_pressure(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    hyperscaler = _basket_relative_return(
        prices, DEFAULT_NARRATIVE_HYPERSCALER_TICKERS, "SPY", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS
    )
    supplier = _basket_relative_return(
        prices, DEFAULT_NARRATIVE_AI_SUPPLIER_TICKERS, "SPY", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS
    )
    news = _news_pressure(
        triage,
        events,
        categories=("ai_unit_economics", "hyperscaler_capex_fcf", "earnings_revision"),
        risk_channels=("hyperscaler_margins", "ai_capex", "margins"),
    )
    pressure = max(0.0, supplier - hyperscaler)
    score = _clip01(0.45 * news + 0.55 * _scale_strength(pressure, DEFAULT_NARRATIVE_SIGNAL_STRONG_RELATIVE_STRENGTH))
    return _Signal(
        signal_id="hyperscaler_capex_fcf_pressure",
        signal_name="Hyperscaler capex / FCF pressure",
        source_threads="AI capex economics and platform-margin commentary",
        data_support="thin_proxy",
        score=score,
        direction="Supplier boom with hyperscaler pressure" if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "Not confirmed",
        evidence=f"Hyperscaler vs SPY {hyperscaler:.1%}; supplier vs SPY {supplier:.1%}; news pressure {news:.2f}.",
        read_through="Tests whether AI capex is rewarding suppliers while pressuring platform-owner free cash flow and margins.",
        data_used="Price divergence and AI unit-economics / earnings-revision news.",
        missing_data="Full Bloomberg-style capex, free cash flow, depreciation, debt issuance, and consensus revision feeds.",
    )


def _ai_inflation_pass_through(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    infra = _basket_relative_return(
        prices, DEFAULT_NARRATIVE_AI_INFRASTRUCTURE_TICKERS, "SPY", DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS
    )
    inflation_proxy = _relative_return(prices, "TIP", "IEF", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    news = _news_pressure(
        triage,
        events,
        categories=("ai_infrastructure", "ai_capex_inflation", "macro_release", "energy_supply"),
        risk_channels=("power_demand", "grid_constraint", "inflation", "semiconductor_demand"),
        text_terms=("memory", "chip shortage", "electricity", "power", "price hike", "inflation"),
    )
    score = _clip01(
        0.45 * news
        + 0.30 * _scale_strength(infra, DEFAULT_NARRATIVE_SIGNAL_RELATIVE_STRENGTH_THRESHOLD)
        + 0.25 * _scale_strength(inflation_proxy, DEFAULT_NARRATIVE_SIGNAL_RELATIVE_STRENGTH_THRESHOLD)
    )
    return _Signal(
        signal_id="ai_capex_inflation_pass_through",
        signal_name="AI capex inflation pass-through",
        source_threads="run-hot policy, AI infrastructure, and inflation-pass-through commentary",
        data_support="thin_proxy",
        score=score,
        direction="AI buildout may be inflationary" if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "Not confirmed",
        evidence=f"AI infrastructure vs SPY {infra:.1%}; TIP/IEF relative {inflation_proxy:.1%}; news pressure {news:.2f}.",
        read_through="Looks for the chain where AI buildout tightens chip/power supply and spills into consumer/device inflation.",
        data_used="Infrastructure price proxies, TIP/IEF inflation proxy, and text/news categories.",
        missing_data="Direct memory contract pricing, device bill-of-material data, utility interconnection queues, and company-specific price actions.",
    )


def _concentration_broadening(prices: pd.DataFrame) -> _Signal:
    qqq_rsp = _relative_return(prices, "QQQ", "RSP", DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS)
    rsp_spy = _relative_return(prices, "RSP", "SPY", DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS)
    iwm_spy = _relative_return(prices, "IWM", "SPY", DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS)
    concentration = max(0.0, qqq_rsp) + max(0.0, -rsp_spy) + max(0.0, -iwm_spy)
    broadening = max(0.0, rsp_spy) + max(0.0, iwm_spy)
    score = _clip01(_scale_strength(concentration - broadening, 0.18))
    direction = "Concentrated leadership" if concentration > broadening else "Broadening/rotation confirmation"
    return _Signal(
        signal_id="concentration_vs_broadening",
        signal_name="Concentration versus broadening",
        source_threads="market concentration, crowding, and broadening commentary",
        data_support="proxy",
        score=score,
        direction=direction,
        evidence=f"QQQ/RSP {qqq_rsp:.1%}; RSP/SPY {rsp_spy:.1%}; IWM/SPY {iwm_spy:.1%}.",
        read_through="Distinguishes a narrow AI-led advance from a healthier broadening market that can justify staying risk-on.",
        data_used="Equal-weight, cap-weight, Nasdaq, and small-cap ETF relative returns.",
        missing_data="Constituent-level contribution, index concentration, and active-manager flow data.",
    )


def _oil_inflation_shock(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    oil = _basket_relative_return(prices, ("USO", "BNO", "XLE", "DBC"), "SPY", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    duration = _relative_return(prices, "TLT", "SPY", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    news = _news_pressure(triage, events, categories=("oil_chokepoint", "energy_supply"))
    score = _clip01(0.55 * news + 0.30 * _scale_strength(oil, 0.10) + 0.15 * _scale_strength(-duration, 0.08))
    return _Signal(
        signal_id="oil_inflation_shock",
        signal_name="Oil / inflation shock",
        source_threads="geopolitical oil, inflation-shock, and weekly market-risk commentary",
        data_support="proxy",
        score=score,
        direction="Oil/inflation pressure active" if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "Contained",
        evidence=f"Oil/commodity basket vs SPY {oil:.1%}; TLT vs SPY {duration:.1%}; news pressure {news:.2f}.",
        read_through="Separates an energy/geopolitical scare from a confirmed inflationary risk-off impulse.",
        data_used="Oil/commodity ETFs, duration ETF, and oil/energy event categories.",
        missing_data="Real-time physical cargo, refinery margin, options skew, and shipping-flow data.",
    )


def _private_credit_liquidity(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    credit = _relative_return(prices, "HYG", "LQD", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    private_proxy = _basket_relative_return(prices, ("BIZD", "SRLN", "BKLN", "JAAA", "JBBB"), "LQD", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    news = _news_pressure(triage, events, categories=("private_credit", "market_plumbing"))
    score = _clip01(0.60 * news + 0.25 * _scale_strength(-credit, 0.05) + 0.15 * _scale_strength(-private_proxy, 0.05))
    return _Signal(
        signal_id="private_credit_liquidity",
        signal_name="Private credit / liquidity stress",
        source_threads="private-credit, market-plumbing, and liquidity-risk commentary",
        data_support="proxy",
        score=score,
        direction="Credit liquidity pressure" if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "Contained",
        evidence=f"HYG/LQD {credit:.1%}; private-credit proxy vs LQD {private_proxy:.1%}; news pressure {news:.2f}.",
        read_through="Checks whether private-market concern is leaking into liquid credit, small caps, or volatility.",
        data_used="Liquid credit ETFs, BDC/loan proxies, and private-credit event/news categories.",
        missing_data="Private fund marks, redemption queues, loan-level covenant data, and dealer balance-sheet data.",
    )


def _policy_put_uncertainty(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    defensive = _basket_relative_return(prices, DEFAULT_NARRATIVE_DEFENSIVE_CONFIRMATION_TICKERS, "SPY", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    news = _news_pressure(triage, events, categories=("monetary_policy", "macro_release", "market_plumbing"))
    score = _clip01(0.65 * news + 0.35 * _scale_strength(defensive, 0.10))
    return _Signal(
        signal_id="fed_put_policy_uncertainty",
        signal_name="Fed-put / policy uncertainty",
        source_threads="policy reaction-function, Fed-put, and rates-risk commentary",
        data_support="thin_proxy",
        score=score,
        direction="Policy uncertainty active" if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "Not confirmed",
        evidence=f"Defensive confirmation basket vs SPY {defensive:.1%}; policy/news pressure {news:.2f}.",
        read_through="Flags when Fed communication, rates, dollar, credit, or funding conditions argue for smaller position sizes.",
        data_used="Monetary-policy/news categories plus TLT/UUP/HYG/LQD/VIXY behavior.",
        missing_data="Fed-speak NLP history, OIS-implied path, dealer rate-vol exposure, and real-time reserve/liquidity data.",
    )


def _speculative_leverage_proxy(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    speculation = _basket_relative_return(prices, DEFAULT_NARRATIVE_SPECULATIVE_TICKERS, "SPY", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    news = _news_pressure(triage, events, categories=("retail_sentiment", "market_plumbing"))
    score = _clip01(0.50 * news + 0.50 * _scale_strength(abs(speculation), 0.12))
    return _Signal(
        signal_id="speculative_leverage_proxy",
        signal_name="Speculative leverage proxy",
        source_threads="crowding, speculative positioning, and leverage-risk commentary",
        data_support="thin_proxy",
        score=score,
        direction="Speculation/crowding pressure" if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "Not confirmed",
        evidence=f"Speculative proxy basket vs SPY {speculation:.1%}; plumbing/retail news pressure {news:.2f}.",
        read_through="Uses public proxies for leverage appetite and squeeze risk while acknowledging true positioning data is missing.",
        data_used="ARKK/SPHB/IWM/IBIT/VIXY behavior plus retail/plumbing news categories.",
        missing_data="Margin debt, dealer gamma, CTA exposure, option-flow, short-interest, and levered ETF AUM feeds.",
    )


def _ipo_equity_supply_pressure(triage: pd.DataFrame, events: tuple[MarketEvent, ...]) -> _Signal:
    news = _news_pressure(
        triage,
        events,
        categories=("equity_supply", "market_plumbing"),
        text_terms=("ipo", "lockup", "secondary offering", "share sale", "convertible", "index inclusion", "free float"),
    )
    return _Signal(
        signal_id="ipo_equity_supply_pressure",
        signal_name="IPO / equity-supply pressure",
        source_threads="equity-supply and market-structure commentary",
        data_support="thin_proxy",
        score=news,
        direction="Equity supply watch active" if news >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "Sparse evidence",
        evidence=f"Equity-supply news pressure {news:.2f}.",
        read_through="Keeps IPOs, secondaries, lockups, index inclusion, and public-float growth visible as market-plumbing risks.",
        data_used="News/event text only.",
        missing_data="IPO calendar, lockup calendar, public-float changes, ETF/index demand estimates, and deal-book data.",
    )


def _positive_catalyst_absorption(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    qqq_5d = _absolute_return(prices, "QQQ", DEFAULT_NARRATIVE_SIGNAL_ABSORPTION_LOOKBACK_DAYS)
    smh_5d = _absolute_return(prices, "SMH", DEFAULT_NARRATIVE_SIGNAL_ABSORPTION_LOOKBACK_DAYS)
    news = _news_pressure(
        triage,
        events,
        categories=("earnings_revision", "ai_infrastructure"),
        directions=("deescalation",),
        text_terms=("beat", "raised guidance", "record", "strong demand", "earnings"),
    )
    failed_absorption = max(0.0, smh_5d - qqq_5d)
    score = _clip01(0.55 * news + 0.45 * _scale_strength(failed_absorption, 0.08))
    return _Signal(
        signal_id="positive_catalyst_absorption",
        signal_name="Positive catalyst absorption",
        source_threads="earnings and positive-catalyst absorption commentary",
        data_support="proxy",
        score=score,
        direction="Good news not lifting broad growth" if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "No failure detected",
        evidence=f"QQQ 5d {qqq_5d:.1%}; SMH 5d {smh_5d:.1%}; positive-catalyst news pressure {news:.2f}.",
        read_through="Detects when strong semiconductor/earnings news fails to pull the broader growth index higher.",
        data_used="Recent QQQ/SMH returns plus earnings/AI infrastructure news text.",
        missing_data="Event-level earnings surprise database and intraday post-earnings reaction data.",
    )


def _international_chip_concentration(prices: pd.DataFrame) -> _Signal:
    chip = _basket_relative_return(prices, DEFAULT_NARRATIVE_GLOBAL_CHIP_TICKERS, "SPY", DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS)
    international = _basket_relative_return(prices, ("EFA", "EEM", "VEA", "VWO", "EWJ"), "SPY", DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS)
    score = _clip01(_scale_strength(chip - international, DEFAULT_NARRATIVE_SIGNAL_STRONG_RELATIVE_STRENGTH))
    return _Signal(
        signal_id="international_chip_concentration",
        signal_name="International chip concentration",
        source_threads="global diversification and chip-concentration commentary",
        data_support="thin_proxy",
        score=score,
        direction="Global performance is chip-concentrated" if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "No clear concentration",
        evidence=f"Global-chip proxy vs SPY {chip:.1%}; international equity proxy vs SPY {international:.1%}.",
        read_through="Checks whether non-US strength is broad diversification or simply another semiconductor concentration trade.",
        data_used="TSM/ASML/SMH/SOXX/MU and broad international ETF proxies.",
        missing_data="Constituent-level regional contribution and local-market Samsung/SK Hynix data.",
    )


def _sector_valuation_policy_proxy(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    healthcare = _relative_return(prices, "XLV", "SPY", DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS)
    policy = _news_pressure(
        triage,
        events,
        categories=("regulatory_filing", "trade_policy", "macro_release"),
        risk_channels=("regulatory", "policy_uncertainty"),
        text_terms=("healthcare", "medicare", "drug pricing", "tariff", "policy risk"),
    )
    score = _clip01(0.55 * policy + 0.45 * _scale_strength(-healthcare, 0.08))
    return _Signal(
        signal_id="sector_valuation_policy_proxy",
        signal_name="Sector valuation / policy proxy",
        source_threads="sector valuation and policy-risk commentary",
        data_support="thin_proxy",
        score=score,
        direction="Cheap/policy-exposed sectors need review" if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE else "No active proxy pressure",
        evidence=f"XLV/SPY {healthcare:.1%}; policy news pressure {policy:.2f}.",
        read_through="Keeps sectors that look cheap but politically exposed from being treated as simple mean-reversion opportunities.",
        data_used="Sector relative performance plus regulatory/policy news categories.",
        missing_data="Sector valuation spreads, earnings revisions, policy probability, and analyst estimate dispersion.",
    )


def _easy_bubble_vs_hard_risk_off(
    prices: pd.DataFrame, triage: pd.DataFrame, events: tuple[MarketEvent, ...]
) -> _Signal:
    ai = _basket_relative_return(prices, DEFAULT_NARRATIVE_AI_SUPPLIER_TICKERS, "SPY", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    credit = _relative_return(prices, "HYG", "LQD", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    vol = _absolute_return(prices, "VIXY", DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS)
    plumbing_news = _news_pressure(triage, events, categories=("market_plumbing", "private_credit"))
    easy = max(0.0, ai) + max(0.0, credit) + max(0.0, -vol)
    hard = max(0.0, -credit) + max(0.0, vol) + plumbing_news
    score = _clip01(_scale_strength(abs(easy - hard), 0.25))
    direction = "Easy-bubble/rotation risk" if easy >= hard else "Hard risk-off pressure"
    return _Signal(
        signal_id="easy_bubble_vs_hard_risk_off",
        signal_name="Easy bubble versus hard risk-off",
        source_threads="late-cycle bubble, rotation, and credit-break commentary",
        data_support="proxy",
        score=score,
        direction=direction,
        evidence=f"AI supplier relative {ai:.1%}; HYG/LQD {credit:.1%}; VIXY 1m {vol:.1%}; plumbing news {plumbing_news:.2f}.",
        read_through="Distinguishes a choppy rotation/bubble market from a true liquidity or credit break that deserves larger cash moves.",
        data_used="AI supplier relative strength, credit relative strength, volatility ETF behavior, and plumbing/private-credit news.",
        missing_data="Dealer positioning, broad fund flows, and leverage/liquidation maps.",
    )


def _unsupported_data_watchlist() -> _Signal:
    return _Signal(
        signal_id="paid_or_unavailable_data_watchlist",
        signal_name="Unsupported data watchlist",
        source_threads="All external commentary sources",
        data_support="unsupported_watchlist",
        score=0.0,
        direction="Do not score as if observed",
        evidence="Useful themes exist, but the project lacks direct data for several institutional feeds.",
        read_through="Treat these as data-acquisition gaps, not active trading signals.",
        data_used="None; this row prevents false precision.",
        missing_data=(
            "Bloomberg-style consensus revisions, dealer/CTA/gamma positioning, ETF/fund flows, "
            "margin debt timeliness, IPO/lockup calendars, constituent contribution, and full capex/FCF feeds."
        ),
        trade_use="Do not trade from this row; use it to prioritize future data sourcing or proxy validation.",
    )


def _news_pressure(
    triage: pd.DataFrame,
    events: tuple[MarketEvent, ...],
    *,
    categories: tuple[str, ...] = (),
    risk_channels: tuple[str, ...] = (),
    directions: tuple[str, ...] = ("escalation", "uncertain", "deescalation"),
    text_terms: tuple[str, ...] = (),
) -> float:
    values: list[float] = []
    category_set = set(categories)
    direction_set = set(directions)
    if not triage.empty:
        for _, row in triage.iterrows():
            category = str(row.get("category", ""))
            direction = str(row.get("direction", ""))
            text = " ".join(
                str(row.get(column, "")).lower() for column in ("title", "summary", "topics", "risk_channels")
            )
            category_match = not category_set or category in category_set
            direction_match = direction in direction_set
            channel_match = not risk_channels or any(channel in text for channel in risk_channels)
            text_match = not text_terms or any(term.lower() in text for term in text_terms)
            if category_match and direction_match and channel_match and text_match:
                urgency = _to_float(row.get("urgency_score"), 0.0)
                values.append(max(urgency, DEFAULT_NARRATIVE_SIGNAL_NEWS_URGENCY_THRESHOLD))
    for event in events:
        event_text = " ".join([event.name, event.description, " ".join(event.tags)]).lower()
        category_match = not category_set or event.category in category_set
        direction_match = event.direction in direction_set
        text_match = not text_terms or any(term.lower() in event_text for term in text_terms)
        if event.current and category_match and direction_match and text_match:
            values.append(0.85 if event.direction == "escalation" else 0.65)
    if not values:
        return 0.0
    return _clip01(float(np.mean(values[:5])))


def _basket_relative_return(
    prices: pd.DataFrame, tickers: Iterable[str], benchmark: str, lookback_days: int
) -> float:
    returns = [
        _relative_return(prices, ticker, benchmark, lookback_days)
        for ticker in tickers
        if ticker in prices.columns and ticker != benchmark
    ]
    clean = [value for value in returns if pd.notna(value)]
    if not clean:
        return 0.0
    return float(np.mean(clean))


def _relative_return(prices: pd.DataFrame, ticker: str, benchmark: str, lookback_days: int) -> float:
    if ticker not in prices.columns or benchmark not in prices.columns:
        return 0.0
    return _absolute_return(prices, ticker, lookback_days) - _absolute_return(
        prices, benchmark, lookback_days
    )


def _absolute_return(prices: pd.DataFrame, ticker: str, lookback_days: int) -> float:
    if ticker not in prices.columns:
        return 0.0
    series = prices[ticker].dropna()
    if len(series) <= lookback_days:
        return 0.0
    start = float(series.iloc[-lookback_days - 1])
    end = float(series.iloc[-1])
    if start <= 0:
        return 0.0
    return end / start - 1.0


def _scale_strength(value: float, target: float) -> float:
    if target <= 0 or pd.isna(value):
        return 0.0
    return _clip01(value / target)


def _clip01(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _status(score: float, data_support: str) -> str:
    if data_support == "unsupported_watchlist":
        return "unsupported_watchlist"
    if score >= DEFAULT_NARRATIVE_SIGNAL_ACTIVE_SCORE:
        return "active"
    if score >= DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE:
        return "warning"
    return "quiet"


def _status_rank(value: object) -> int:
    return {
        "active": 0,
        "warning": 1,
        "quiet": 2,
        "unsupported_watchlist": 3,
    }.get(str(value), 4)


def _to_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed
