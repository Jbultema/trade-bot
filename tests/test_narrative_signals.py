from __future__ import annotations

import pandas as pd

from trade_bot.research.narrative_signals import (
    build_narrative_signal_table,
    summarize_narrative_signals,
)
from trade_bot.research.news_monitor import NewsItem, triage_news_items


def test_narrative_signals_score_proxy_backed_ai_supplier_divergence() -> None:
    index = pd.bdate_range("2026-01-01", periods=90)
    base = pd.Series(range(90), index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "SPY": 100.0 + base * 0.10,
            "SMH": 100.0 + base * 0.80,
            "SOXX": 100.0 + base * 0.75,
            "MU": 100.0 + base * 1.20,
            "MSFT": 100.0 - base * 0.05,
            "AMZN": 100.0 - base * 0.04,
            "GOOGL": 100.0 - base * 0.03,
            "QQQ": 100.0 + base * 0.05,
            "RSP": 100.0 + base * 0.02,
            "IWM": 100.0 + base * 0.02,
            "HYG": 100.0 + base * 0.03,
            "LQD": 100.0 + base * 0.02,
            "VIXY": 100.0 - base * 0.30,
            "UUP": 100.0,
            "TLT": 100.0,
            "TIP": 100.0,
            "IEF": 100.0,
            "XLV": 100.0,
        },
        index=index,
    )
    now = pd.Timestamp(index[-1]).tz_localize("UTC")
    items = (
        NewsItem(
            source="test",
            source_url="https://example.com/rss",
            source_priority=5,
            title="AI infrastructure suppliers surge while hyperscaler capex weighs on free cash flow",
            summary="Data center capex and free cash flow pressure are being debated.",
            url="https://example.com/ai",
            published_at=now.isoformat(),
            topics=("ai", "ai_capex"),
        ),
    )
    triage = triage_news_items(items, lookback_days=7, now=now)

    signals = build_narrative_signal_table(prices, news_triage=triage)
    ai_signal = signals.set_index("signal_id").loc["ai_supplier_hyperscaler_divergence"]
    unsupported = signals.set_index("signal_id").loc["paid_or_unavailable_data_watchlist"]

    assert ai_signal["data_support"] == "proxy"
    assert ai_signal["decision_role"] == "explainer_research_only"
    assert ai_signal["model_authority"] == "no_direct_sizing_authority"
    assert "ablation" in str(ai_signal["promotion_requirement"])
    assert ai_signal["status"] in {"active", "warning"}
    assert float(ai_signal["score"]) > 0.45
    assert unsupported["status"] == "unsupported_watchlist"
    assert unsupported["score"] == 0.0
    assert unsupported["model_authority"] == "no_direct_sizing_authority"


def test_narrative_signal_summary_names_active_supported_themes() -> None:
    signals = pd.DataFrame(
        [
            {
                "signal_id": "a",
                "signal_name": "Proxy theme",
                "source_threads": "demo",
                "data_support": "proxy",
                "score": 0.8,
                "status": "active",
                "direction": "demo",
                "evidence": "demo",
                "read_through": "demo",
                "decision_role": "explainer_research_only",
                "model_authority": "no_direct_sizing_authority",
                "promotion_requirement": "demo",
                "data_used": "demo",
                "missing_data": "demo",
                "trade_use": "demo",
            },
            {
                "signal_id": "b",
                "signal_name": "Thin proxy theme",
                "source_threads": "demo",
                "data_support": "thin_proxy",
                "score": 0.9,
                "status": "active",
                "direction": "demo",
                "evidence": "demo",
                "read_through": "demo",
                "decision_role": "explainer_research_only",
                "model_authority": "no_direct_sizing_authority",
                "promotion_requirement": "demo",
                "data_used": "demo",
                "missing_data": "demo",
                "trade_use": "demo",
            },
            {
                "signal_id": "c",
                "signal_name": "Unsupported theme",
                "source_threads": "demo",
                "data_support": "unsupported_watchlist",
                "score": 0.0,
                "status": "unsupported_watchlist",
                "direction": "demo",
                "evidence": "demo",
                "read_through": "demo",
                "decision_role": "explainer_research_only",
                "model_authority": "no_direct_sizing_authority",
                "promotion_requirement": "demo",
                "data_used": "demo",
                "missing_data": "demo",
                "trade_use": "demo",
            },
        ]
    )

    summary = summarize_narrative_signals(signals)

    assert summary["answer"] == "1 active narrative signal(s)"
    assert "Proxy theme" in summary["detail"]
    assert "Thin proxy theme" not in summary["detail"]
    assert "Unsupported theme" not in summary["detail"]
    assert "Research-only thin proxies: 1" in summary["detail"]
    assert "not a direct sizing input" in summary["detail"]
