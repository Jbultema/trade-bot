from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree

import pandas as pd
import requests
import yaml

from trade_bot.DEFAULTS import (
    DEFAULT_NEWS_ACTIVATION_THRESHOLD,
    DEFAULT_NEWS_CACHE_FILE,
    DEFAULT_NEWS_LOOKBACK_DAYS,
    DEFAULT_NEWS_MAX_AGE_MINUTES,
    DEFAULT_NEWS_MAX_ITEMS_PER_SOURCE,
    DEFAULT_NEWS_SOURCE_COVERAGE_BUCKETS,
    DEFAULT_NEWS_SOURCE_ENABLED,
    DEFAULT_NEWS_SOURCE_PRIORITY,
    DEFAULT_NEWS_SOURCE_TYPE,
    DEFAULT_NEWS_USER_AGENT,
)
from trade_bot.research.event_risk import (
    EventDirection,
    MarketEvent,
    NewsPhase,
    classify_news_text,
)

TRIAGE_COLUMNS = [
    "title",
    "source",
    "published_at",
    "category",
    "direction",
    "phase",
    "urgency_score",
    "activation_status",
    "event_id",
    "confidence",
    "source_priority",
    "risk_channels",
    "candidate_proxies",
    "tradable_question",
    "confirmation_window",
    "url",
    "summary",
]
SOURCE_HEALTH_COLUMNS = ["source", "status", "fetched_at", "items", "message", "url"]
SOURCE_COVERAGE_COLUMNS = [
    "coverage_bucket",
    "required_topics",
    "source_count",
    "enabled_source_count",
    "max_priority",
    "status",
    "source_names",
]


@dataclass(frozen=True)
class NewsSource:
    name: str
    url: str
    source_type: str = DEFAULT_NEWS_SOURCE_TYPE
    topics: tuple[str, ...] = ()
    priority: int = DEFAULT_NEWS_SOURCE_PRIORITY
    enabled: bool = DEFAULT_NEWS_SOURCE_ENABLED


@dataclass(frozen=True)
class NewsConfig:
    sources: tuple[NewsSource, ...]
    max_age_minutes: int = DEFAULT_NEWS_MAX_AGE_MINUTES
    lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS
    activation_threshold: float = DEFAULT_NEWS_ACTIVATION_THRESHOLD
    max_items_per_source: int = DEFAULT_NEWS_MAX_ITEMS_PER_SOURCE


@dataclass(frozen=True)
class NewsItem:
    source: str
    source_url: str
    source_priority: int
    title: str
    summary: str
    url: str
    published_at: str | None
    topics: tuple[str, ...] = ()


@dataclass(frozen=True)
class NewsMonitorRun:
    items: tuple[NewsItem, ...]
    triage: pd.DataFrame
    source_health: pd.DataFrame
    activated_events: tuple[MarketEvent, ...]
    activation_threshold: float
    lookback_days: int
    source_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_news_config(path: str | Path | None) -> NewsConfig:
    if path is None:
        return NewsConfig(sources=())
    config_path = Path(path)
    if not config_path.exists():
        return NewsConfig(sources=())

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    settings = raw.get("news", {})
    sources = tuple(_news_source_from_mapping(item) for item in raw.get("sources", []))
    return NewsConfig(
        sources=sources,
        max_age_minutes=int(settings.get("max_age_minutes", DEFAULT_NEWS_MAX_AGE_MINUTES)),
        lookback_days=int(settings.get("lookback_days", DEFAULT_NEWS_LOOKBACK_DAYS)),
        activation_threshold=float(
            settings.get("activation_threshold", DEFAULT_NEWS_ACTIVATION_THRESHOLD)
        ),
        max_items_per_source=int(
            settings.get("max_items_per_source", DEFAULT_NEWS_MAX_ITEMS_PER_SOURCE)
        ),
    )


def run_news_monitor(
    config_path: str | Path | None,
    *,
    cache_dir: str | Path,
    refresh: bool = False,
    now: pd.Timestamp | None = None,
) -> NewsMonitorRun:
    config = load_news_config(config_path)
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    if not config.sources:
        return _empty_news_monitor(config)

    cache_path = Path(cache_dir) / DEFAULT_NEWS_CACHE_FILE
    cached = _load_cache(cache_path)
    if not refresh and _cache_is_fresh(cached, config.max_age_minutes, now):
        items, health = _items_and_health_from_cache(cached)
    else:
        items, health = fetch_news_sources(
            config.sources, max_items_per_source=config.max_items_per_source
        )
        if not items and cached:
            cached_items, cached_health = _items_and_health_from_cache(cached)
            items = cached_items
            health = _cache_fallback_health(cached_health, now)
        elif items:
            _write_cache(cache_path, items, health, now)

    triage = triage_news_items(items, lookback_days=config.lookback_days, now=now)
    return NewsMonitorRun(
        items=items,
        triage=triage,
        source_health=health,
        activated_events=(),
        activation_threshold=config.activation_threshold,
        lookback_days=config.lookback_days,
        source_coverage=build_news_source_coverage(config.sources),
    )


def build_news_source_coverage(sources: tuple[NewsSource, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket, required_topics in DEFAULT_NEWS_SOURCE_COVERAGE_BUCKETS.items():
        matched_sources = [
            source for source in sources if set(source.topics).intersection(required_topics)
        ]
        enabled_sources = [source for source in matched_sources if source.enabled]
        max_priority = max((source.priority for source in enabled_sources), default=0)
        if len(enabled_sources) >= 2 and max_priority >= 3:
            status = "covered"
        elif enabled_sources:
            status = "thin"
        else:
            status = "blind_spot"
        rows.append(
            {
                "coverage_bucket": bucket,
                "required_topics": ", ".join(required_topics),
                "source_count": len(matched_sources),
                "enabled_source_count": len(enabled_sources),
                "max_priority": max_priority,
                "status": status,
                "source_names": ", ".join(source.name for source in enabled_sources),
            }
        )
    return pd.DataFrame(rows, columns=SOURCE_COVERAGE_COLUMNS)


def fetch_news_sources(
    sources: tuple[NewsSource, ...],
    *,
    max_items_per_source: int,
) -> tuple[tuple[NewsItem, ...], pd.DataFrame]:
    items: list[NewsItem] = []
    health_rows: list[dict[str, object]] = []
    fetched_at = pd.Timestamp.now(tz="UTC").isoformat()

    for source in sources:
        if not source.enabled:
            continue
        if source.source_type != "rss":
            health_rows.append(
                _source_health_row(source, "skipped", fetched_at, 0, "Unsupported source type.")
            )
            continue
        try:
            response = requests.get(
                source.url,
                timeout=12,
                headers={"User-Agent": DEFAULT_NEWS_USER_AGENT},
            )
            response.raise_for_status()
            source_items = _parse_feed(_xml_response_text(response), source)[:max_items_per_source]
            items.extend(source_items)
            health_rows.append(
                _source_health_row(source, "ok", fetched_at, len(source_items), "Fetched.")
            )
        except Exception as exc:  # pragma: no cover - exercised in integration runs.
            health_rows.append(_source_health_row(source, "error", fetched_at, 0, str(exc)))

    unique_items = _dedupe_news_items(items)
    return tuple(unique_items), pd.DataFrame(health_rows, columns=SOURCE_HEALTH_COLUMNS)


def triage_news_items(
    items: tuple[NewsItem, ...],
    *,
    lookback_days: int,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if now is None:
        now = pd.Timestamp.now(tz="UTC")

    rows: list[dict[str, object]] = []
    min_date = now - pd.Timedelta(days=lookback_days)
    for item in items:
        published_at = _parse_timestamp(item.published_at)
        if published_at is not None and published_at < min_date:
            continue

        classification = classify_news_text(f"{item.title}. {item.summary}")
        score = _urgency_score(item, classification, published_at, now)
        rows.append(
            {
                "title": item.title,
                "source": item.source,
                "published_at": published_at.isoformat() if published_at is not None else "",
                "category": classification.category,
                "direction": classification.direction,
                "phase": classification.phase,
                "urgency_score": score,
                "activation_status": "pending",
                "event_id": "",
                "confidence": classification.confidence,
                "source_priority": item.source_priority,
                "risk_channels": ", ".join(classification.risk_channels),
                "candidate_proxies": ", ".join(classification.candidate_proxies),
                "tradable_question": classification.tradable_question,
                "confirmation_window": classification.confirmation_window,
                "phase_reason": classification.phase_reason,
                "url": item.url,
                "summary": item.summary,
                "topics": ", ".join(item.topics),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[*TRIAGE_COLUMNS, "phase_reason", "topics"])

    triage = pd.DataFrame(rows)
    return triage.sort_values(
        ["urgency_score", "published_at"], ascending=[False, False]
    ).reset_index(drop=True)


def activate_news_events(
    news_monitor: NewsMonitorRun,
    existing_events: tuple[MarketEvent, ...],
) -> NewsMonitorRun:
    if news_monitor.triage.empty:
        return news_monitor

    existing_by_url = {
        _canonical_url(str(event.source_url)): event
        for event in existing_events
        if event.source_url
    }
    activated_events: list[MarketEvent] = []
    rows: list[dict[str, object]] = []
    used_urls: set[str] = set()

    for row in news_monitor.triage.to_dict("records"):
        mutable_row = dict(row)
        canonical_url = _canonical_url(str(row["url"]))
        category = str(row["category"])
        urgency_score = float(row["urgency_score"])
        source_priority = int(row["source_priority"])

        if canonical_url in existing_by_url:
            event = existing_by_url[canonical_url]
            mutable_row["activation_status"] = "covered_by_curated_event"
            mutable_row["event_id"] = event.event_id
            used_urls.add(canonical_url)
        elif (
            category != "unclassified"
            and urgency_score >= news_monitor.activation_threshold
            and source_priority >= 3
        ):
            event = _market_event_from_news_row(mutable_row)
            mutable_row["activation_status"] = "event_risk_generated"
            mutable_row["event_id"] = event.event_id
            activated_events.append(event)
            used_urls.add(canonical_url)
        elif category != "unclassified" and urgency_score >= news_monitor.activation_threshold:
            mutable_row["activation_status"] = "triage_only_low_priority"
        elif category != "unclassified":
            mutable_row["activation_status"] = "triage_only_below_threshold"
        else:
            mutable_row["activation_status"] = "unclassified"
        rows.append(mutable_row)

    triage = pd.DataFrame(rows)
    triage = triage.sort_values(
        ["urgency_score", "published_at"], ascending=[False, False]
    ).reset_index(drop=True)
    return NewsMonitorRun(
        items=news_monitor.items,
        triage=triage,
        source_health=news_monitor.source_health,
        activated_events=tuple(activated_events),
        activation_threshold=news_monitor.activation_threshold,
        lookback_days=news_monitor.lookback_days,
        source_coverage=news_monitor.source_coverage,
    )


def _news_source_from_mapping(raw: dict[str, Any]) -> NewsSource:
    return NewsSource(
        name=str(raw["name"]),
        url=str(raw["url"]),
        source_type=str(raw.get("source_type", "rss")),
        topics=tuple(str(topic) for topic in raw.get("topics", [])),
        priority=int(raw.get("priority", 3)),
        enabled=bool(raw.get("enabled", True)),
    )


def _xml_response_text(response: requests.Response) -> str:
    return response.content.decode("utf-8-sig", errors="replace").lstrip()


def _parse_feed(xml_text: str, source: NewsSource) -> list[NewsItem]:
    root = ElementTree.fromstring(xml_text.encode("utf-8"))
    root_name = _local_name(root.tag)
    if root_name == "rss":
        channel = _first_child(root, "channel")
        if channel is None:
            return []
        return [_rss_item_to_news_item(item, source) for item in _children(channel, "item")]
    if root_name == "feed":
        return [_atom_entry_to_news_item(entry, source) for entry in _children(root, "entry")]
    return []


def _rss_item_to_news_item(item: ElementTree.Element, source: NewsSource) -> NewsItem:
    title = _clean_text(_first_text(item, "title"))
    summary = _clean_text(
        _first_text(item, "description")
        or _first_text(item, "summary")
        or _first_text(item, "encoded")
    )
    url = _first_text(item, "link") or _first_text(item, "guid")
    published_at = _normalize_datetime_text(
        _first_text(item, "pubDate") or _first_text(item, "published")
    )
    return NewsItem(
        source=source.name,
        source_url=source.url,
        source_priority=source.priority,
        title=title,
        summary=summary,
        url=url.strip(),
        published_at=published_at,
        topics=source.topics,
    )


def _atom_entry_to_news_item(entry: ElementTree.Element, source: NewsSource) -> NewsItem:
    title = _clean_text(_first_text(entry, "title"))
    summary = _clean_text(_first_text(entry, "summary") or _first_text(entry, "content"))
    url = _atom_link(entry)
    published_at = _normalize_datetime_text(
        _first_text(entry, "published") or _first_text(entry, "updated")
    )
    return NewsItem(
        source=source.name,
        source_url=source.url,
        source_priority=source.priority,
        title=title,
        summary=summary,
        url=url.strip(),
        published_at=published_at,
        topics=source.topics,
    )


def _atom_link(entry: ElementTree.Element) -> str:
    links = _children(entry, "link")
    for link in links:
        if link.attrib.get("rel", "alternate") == "alternate" and link.attrib.get("href"):
            return str(link.attrib["href"])
    if links and links[0].attrib.get("href"):
        return str(links[0].attrib["href"])
    return ""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _first_child(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    children = _children(element, name)
    if not children:
        return None
    return children[0]


def _first_text(element: ElementTree.Element, *names: str) -> str:
    for name in names:
        child = _first_child(element, name)
        if child is not None and child.text:
            return child.text
    return ""


def _clean_text(text: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", unescape(no_tags)).strip()
    return clean


def _normalize_datetime_text(value: str) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).astimezone().isoformat()
    except (TypeError, ValueError, AttributeError):
        try:
            return pd.Timestamp(value).isoformat()
        except ValueError:
            return None


def _parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _urgency_score(
    item: NewsItem,
    classification: Any,
    published_at: pd.Timestamp | None,
    now: pd.Timestamp,
) -> float:
    confidence_component = float(classification.confidence) * 0.35
    source_component = min(max(item.source_priority, 1), 5) / 5.0 * 0.15
    category_component = _category_weight(str(classification.category))
    phase_component = _phase_weight(classification.phase)
    direction_component = _direction_weight(classification.direction)
    recency_component = _recency_weight(published_at, now)
    score = (
        confidence_component
        + source_component
        + category_component
        + phase_component
        + direction_component
        + recency_component
    )
    return round(min(max(score, 0.0), 1.0), 4)


def _category_weight(category: str) -> float:
    weights = {
        "ai_unit_economics": 0.18,
        "ai_infrastructure": 0.11,
        "private_credit": 0.18,
        "oil_chokepoint": 0.16,
        "energy_supply": 0.12,
        "trade_policy": 0.15,
        "military_escalation": 0.13,
        "monetary_policy": 0.17,
        "macro_release": 0.14,
        "earnings_revision": 0.15,
        "market_plumbing": 0.18,
        "regulatory_filing": 0.13,
        "retail_sentiment": 0.08,
    }
    return weights.get(category, 0.0)


def _phase_weight(phase: NewsPhase) -> float:
    weights = {
        "leading_warning": 0.18,
        "coincident_confirmation": 0.12,
        "phase_uncertain": 0.04,
        "lagging_explanation": -0.05,
    }
    return weights[phase]


def _direction_weight(direction: EventDirection) -> float:
    weights = {
        "escalation": 0.10,
        "uncertain": 0.03,
        "deescalation": 0.02,
    }
    return weights[direction]


def _recency_weight(published_at: pd.Timestamp | None, now: pd.Timestamp) -> float:
    if published_at is None:
        return 0.02
    age = now - published_at
    if age <= pd.Timedelta(days=1):
        return 0.10
    if age <= pd.Timedelta(days=3):
        return 0.07
    if age <= pd.Timedelta(days=7):
        return 0.04
    return 0.0


def _market_event_from_news_row(row: dict[str, object]) -> MarketEvent:
    url = str(row["url"])
    event_id = f"news_{hashlib.sha1(_canonical_url(url).encode('utf-8')).hexdigest()[:12]}"
    event_date = _event_date_from_row(row)
    tags = tuple(
        item.strip()
        for value in (str(row.get("topics", "")), str(row.get("risk_channels", "")))
        for item in value.split(",")
        if item.strip()
    )
    return MarketEvent(
        event_id=event_id,
        name=str(row["title"])[:160],
        date=event_date,
        category=str(row["category"]),
        direction=_direction_from_row(row),
        description=str(row.get("summary", "")),
        source_url=url,
        tags=tags,
        current=True,
        phase=_phase_from_row(row),
        phase_reason=str(row.get("phase_reason", "")),
        confirmation_window=str(row.get("confirmation_window", "")),
    )


def _event_date_from_row(row: dict[str, object]) -> pd.Timestamp:
    published_at = _parse_timestamp(str(row.get("published_at", "")))
    if published_at is None:
        published_at = pd.Timestamp.now(tz="UTC")
    return published_at.tz_convert("UTC").tz_localize(None).normalize()


def _direction_from_row(row: dict[str, object]) -> EventDirection:
    value = str(row["direction"])
    if value in {"escalation", "deescalation", "uncertain"}:
        return value  # type: ignore[return-value]
    return "uncertain"


def _phase_from_row(row: dict[str, object]) -> NewsPhase:
    value = str(row["phase"])
    if value in {
        "leading_warning",
        "coincident_confirmation",
        "lagging_explanation",
        "phase_uncertain",
    }:
        return value  # type: ignore[return-value]
    return "phase_uncertain"


def _dedupe_news_items(items: list[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        key = _canonical_url(item.url) or f"{item.source}:{item.title}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _canonical_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlsplit(url.strip())
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def _source_health_row(
    source: NewsSource,
    status: str,
    fetched_at: str,
    items: int,
    message: str,
) -> dict[str, object]:
    return {
        "source": source.name,
        "status": status,
        "fetched_at": fetched_at,
        "items": items,
        "message": message,
        "url": source.url,
    }


def _cache_fallback_health(health: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if health.empty:
        return health
    fallback = health.copy()
    fallback["status"] = "cache_fallback"
    fallback["fetched_at"] = now.isoformat()
    fallback["message"] = "Using stale cached news because live fetch returned no items."
    return fallback


def _cache_is_fresh(cached: dict[str, Any] | None, max_age_minutes: int, now: pd.Timestamp) -> bool:
    if not cached:
        return False
    fetched_at = _parse_timestamp(str(cached.get("fetched_at", "")))
    if fetched_at is None:
        return False
    return now - fetched_at <= pd.Timedelta(minutes=max_age_minutes)


def _load_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_cache(
    path: Path,
    items: tuple[NewsItem, ...],
    health: pd.DataFrame,
    now: pd.Timestamp,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": now.isoformat(),
        "items": [asdict(item) for item in items],
        "source_health": health.to_dict("records"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _items_and_health_from_cache(
    cached: dict[str, Any] | None,
) -> tuple[tuple[NewsItem, ...], pd.DataFrame]:
    if not cached:
        return (), pd.DataFrame(columns=SOURCE_HEALTH_COLUMNS)
    items = tuple(
        NewsItem(
            source=str(raw["source"]),
            source_url=str(raw["source_url"]),
            source_priority=int(raw["source_priority"]),
            title=str(raw["title"]),
            summary=str(raw.get("summary", "")),
            url=str(raw["url"]),
            published_at=raw.get("published_at"),
            topics=tuple(str(topic) for topic in raw.get("topics", [])),
        )
        for raw in cached.get("items", [])
    )
    health = pd.DataFrame(cached.get("source_health", []), columns=SOURCE_HEALTH_COLUMNS)
    return items, health


def _empty_news_monitor(config: NewsConfig) -> NewsMonitorRun:
    return NewsMonitorRun(
        items=(),
        triage=pd.DataFrame(columns=[*TRIAGE_COLUMNS, "phase_reason", "topics"]),
        source_health=pd.DataFrame(columns=SOURCE_HEALTH_COLUMNS),
        activated_events=(),
        activation_threshold=config.activation_threshold,
        lookback_days=config.lookback_days,
        source_coverage=build_news_source_coverage(config.sources),
    )
