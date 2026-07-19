from __future__ import annotations

import pandas as pd

from trade_bot.research.event_risk import MarketEvent
from trade_bot.research.news_monitor import (
    NewsItem,
    NewsMonitorRun,
    activate_news_events,
    build_news_source_coverage,
    load_news_config,
    triage_news_items,
)


def test_news_triage_promotes_openai_financials_to_event_risk() -> None:
    now = pd.Timestamp("2026-06-17T18:00:00Z")
    items = (
        NewsItem(
            source="Where's Your Ed",
            source_url="https://www.wheresyoured.at/rss/",
            source_priority=5,
            title="Exclusive: OpenAI losses increased nearly 8x in 2025",
            summary="Audited financials showed revenue growth, AI capex spending, losses, and compute costs.",
            url="https://www.wheresyoured.at/exclusive-openai-financials/",
            published_at="2026-06-15T14:00:00Z",
            topics=("ai", "ai_capex"),
        ),
    )
    triage = triage_news_items(items, lookback_days=7, now=now)
    monitor = NewsMonitorRun(
        items=items,
        triage=triage,
        source_health=pd.DataFrame(),
        activated_events=(),
        activation_threshold=0.68,
        lookback_days=7,
    )

    activated = activate_news_events(monitor, existing_events=())

    assert len(activated.activated_events) == 1
    assert activated.activated_events[0].category == "ai_unit_economics"
    assert activated.activated_events[0].phase == "leading_warning"
    assert activated.triage.iloc[0]["activation_status"] == "event_risk_generated"


def test_news_activation_marks_curated_duplicate_as_covered() -> None:
    now = pd.Timestamp("2026-06-17T18:00:00Z")
    url = "https://www.wheresyoured.at/exclusive-openai-financials/"
    items = (
        NewsItem(
            source="Where's Your Ed",
            source_url="https://www.wheresyoured.at/rss/",
            source_priority=5,
            title="Exclusive audited OpenAI financials show losses",
            summary="OpenAI revenue, losses, spending, and AI capex costs were revealed.",
            url=f"{url}?utm_source=rss",
            published_at="2026-06-15T14:00:00Z",
            topics=("ai",),
        ),
    )
    triage = triage_news_items(items, lookback_days=7, now=now)
    monitor = NewsMonitorRun(
        items=items,
        triage=triage,
        source_health=pd.DataFrame(),
        activated_events=(),
        activation_threshold=0.68,
        lookback_days=7,
    )
    curated_event = MarketEvent(
        event_id="openai_financials_zitron_2026_06_15",
        name="OpenAI audited financials raise AI unit-economics concern",
        date=pd.Timestamp("2026-06-15"),
        category="ai_unit_economics",
        direction="escalation",
        description="test",
        source_url=url,
        current=True,
    )

    activated = activate_news_events(monitor, existing_events=(curated_event,))

    assert activated.activated_events == ()
    assert activated.triage.iloc[0]["activation_status"] == "covered_by_curated_event"
    assert activated.triage.iloc[0]["event_id"] == curated_event.event_id


def test_news_activation_keeps_low_priority_sources_in_triage_only() -> None:
    now = pd.Timestamp("2026-06-17T18:00:00Z")
    items = (
        NewsItem(
            source="Reddit Investing",
            source_url="https://www.reddit.com/r/investing/.rss",
            source_priority=2,
            title="Exclusive audited OpenAI financials show losses",
            summary="OpenAI revenue, losses, spending, and AI capex costs were revealed.",
            url="https://www.reddit.com/r/investing/comments/demo",
            published_at="2026-06-17T14:00:00Z",
            topics=("retail_sentiment",),
        ),
    )
    triage = triage_news_items(items, lookback_days=7, now=now)
    monitor = NewsMonitorRun(
        items=items,
        triage=triage,
        source_health=pd.DataFrame(),
        activated_events=(),
        activation_threshold=0.68,
        lookback_days=7,
    )

    activated = activate_news_events(monitor, existing_events=())

    assert activated.activated_events == ()
    assert activated.triage.iloc[0]["activation_status"] == "triage_only_low_priority"


def test_news_triage_excludes_items_published_after_as_of_time() -> None:
    now = pd.Timestamp("2026-06-17T18:00:00Z")
    items = (
        NewsItem(
            source="Future Feed",
            source_url="https://example.com/rss",
            source_priority=5,
            title="OpenAI losses and AI capex pressure",
            summary="AI capex, compute costs, and losses are rising.",
            url="https://example.com/future",
            published_at="2026-06-18T14:00:00Z",
            topics=("ai",),
        ),
    )

    triage = triage_news_items(items, lookback_days=7, now=now)

    assert triage.empty


def test_news_source_config_covers_required_narrative_buckets() -> None:
    config = load_news_config("configs/news_sources.yaml")
    coverage = build_news_source_coverage(config.sources)

    assert set(coverage["status"]) <= {"covered", "thin"}
    assert not coverage[coverage["status"] == "blind_spot"].shape[0]
    assert set(coverage["coverage_bucket"]) >= {
        "official_macro_releases",
        "monetary_policy_liquidity",
        "earnings_revisions_fundamentals",
        "regulatory_filings_enforcement",
        "market_plumbing_volatility",
        "retail_social_sentiment",
    }


def test_news_source_coverage_flags_missing_bucket() -> None:
    config = load_news_config("configs/news_sources.yaml")
    macro_only_sources = tuple(
        source for source in config.sources if "official_macro" in source.topics
    )

    coverage = build_news_source_coverage(macro_only_sources)

    assert "blind_spot" in set(coverage["status"])
    assert coverage.loc[coverage["coverage_bucket"] == "official_macro_releases", "status"].iloc[
        0
    ] in {"covered", "thin"}
