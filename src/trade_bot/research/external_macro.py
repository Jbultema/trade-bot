from __future__ import annotations

import json
import re
import signal
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from trade_bot.storage.warehouse import TradingWarehouse

YOUTUBE_BROWSE_URL = "https://www.youtube.com/youtubei/v1/browse"
DEFAULT_42MACRO_HANDLE = "@42Macro"
DEFAULT_42MACRO_SOURCE = "42macro_youtube"


@dataclass(frozen=True)
class MacroTranscriptSyncResult:
    videos: pd.DataFrame
    fetched: int
    skipped: int
    failed: int
    transcript_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class MacroAlignmentResult:
    videos: pd.DataFrame
    classifications: pd.DataFrame
    comparisons: pd.DataFrame
    summary: dict[str, object]
    output_dir: Path


BULLISH_TERMS: dict[str, float] = {
    "risk-on market regime": 3.0,
    "bullish outlook": 2.5,
    "bull market": 2.0,
    "bubble": 1.7,
    "melt up": 1.7,
    "buy the dip": 1.6,
    "dip should eventually be bought": 1.6,
    "tailwind": 1.1,
    "liquidity": 0.8,
    "easing": 0.8,
    "runway": 0.8,
    "upside": 0.8,
    "support the rally": 0.8,
}

DEFENSIVE_TERMS: dict[str, float] = {
    "risk-off market regime": 3.2,
    "rising probability of a risk-off": 3.0,
    "reduce their gross exposure": 3.0,
    "reduce gross exposure": 3.0,
    "take some chips off": 2.6,
    "book gains": 2.5,
    "source of funds": 2.2,
    "sell-the-news": 2.2,
    "bear market": 2.1,
    "correction": 1.7,
    "volatility": 1.3,
    "hawkish": 1.2,
    "tighten": 1.2,
    "inflationary supply shock": 1.2,
    "risk-off": 1.2,
    "headwind": 1.1,
    "stress": 1.0,
}

LARGE_CHANGE_TERMS = (
    "risk-off market regime",
    "rising probability of a risk-off",
    "reduce gross exposure",
    "reduce their gross exposure",
    "take some chips off",
    "book gains",
    "source of funds",
    "sell-the-news",
    "bear market",
    "mispriced",
    "liquidity crisis",
    "wwiii",
    "strait of hormuz",
    "volatility",
    "crash",
    "bubble",
)

THEME_TERMS = {
    "ai": ("ai", "nvidia", "mag-7", "mag 7", "semiconductor", "chips"),
    "fed_policy": ("fed", "warsh", "powell", "tighten", "easing", "rate"),
    "inflation": ("inflation", "oil", "energy", "supply shock", "tariff"),
    "liquidity": ("liquidity", "treasury", "fiscal", "deficit"),
    "geopolitical": ("strait of hormuz", "iran", "war", "wwiii"),
    "gold": ("gold", "gld"),
    "credit": ("credit", "spreads", "financing"),
}


def sync_42macro_transcripts(
    *,
    transcript_dir: str | Path,
    max_videos: int | None = 250,
    max_pages: int | None = 25,
    refresh: bool = False,
    channel_handle: str = DEFAULT_42MACRO_HANDLE,
    source: str = DEFAULT_42MACRO_SOURCE,
    pause_seconds: float = 0.05,
    transcript_timeout_seconds: int = 30,
    checkpoint_manifest_every: int = 10,
    fetch_transcripts: bool = True,
) -> MacroTranscriptSyncResult:
    """Fetch public 42 Macro YouTube metadata and transcripts into local files."""

    output_dir = Path(transcript_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    catalog = _fetch_channel_video_catalog(
        session=session,
        channel_handle=channel_handle,
        max_videos=max_videos,
        max_pages=max_pages,
    )
    prior_manifest = _load_manifest(output_dir / "manifest.json")
    prior_by_id = {str(row.get("video_id")): row for row in prior_manifest}
    rows: list[dict[str, object]] = []
    fetched = 0
    skipped = 0
    failed = 0

    for video in catalog:
        video_id = str(video["video_id"])
        prior = prior_by_id.get(video_id, {})
        publish_date = str(prior.get("published_date") or prior.get("date") or "")
        if not publish_date:
            publish_date = _fetch_video_publish_date(session, video_id) or ""
        transcript_path = str(prior.get("transcript_path") or prior.get("path") or "")
        existing_path = _resolve_transcript_path(output_dir, transcript_path)
        if (
            existing_path is not None
            and existing_path.exists()
            and not refresh
            and publish_date
        ):
            word_count = _word_count(existing_path.read_text(encoding="utf-8", errors="ignore"))
            status = "skipped_existing"
            error = ""
            transcript_file = existing_path
            skipped += 1
        elif not fetch_transcripts:
            transcript_file = Path("")
            word_count = 0
            status = "catalog_only"
            error = ""
            skipped += 1
        else:
            try:
                transcript = _fetch_youtube_transcript(
                    video_id,
                    timeout_seconds=transcript_timeout_seconds,
                )
                if not publish_date:
                    publish_date = "unknown-date"
                transcript_file = output_dir / _transcript_filename(
                    publish_date,
                    video_id,
                    str(video.get("title", "")),
                )
                transcript_file.write_text(
                    _format_transcript_file(
                        source=source,
                        published_date=publish_date,
                        video_id=video_id,
                        title=str(video.get("title", "")),
                        url=_youtube_url(video_id),
                        transcript=transcript,
                    ),
                    encoding="utf-8",
                )
                word_count = _word_count(transcript)
                status = "fetched"
                error = ""
                fetched += 1
            except Exception as exc:  # pragma: no cover - network/API dependent
                transcript_file = existing_path or Path("")
                word_count = 0
                status = "failed"
                error = str(exc)
                failed += 1
        rows.append(
            {
                "video_id": video_id,
                "source": source,
                "published_date": publish_date,
                "title": str(video.get("title", "")),
                "url": _youtube_url(video_id),
                "transcript_path": _relative_transcript_path(output_dir, transcript_file),
                "word_count": word_count,
                "fetched_at_utc": _utc_now_iso(),
                "status": status,
                "error": error,
            }
        )
        if pause_seconds > 0:
            time.sleep(pause_seconds)
        if checkpoint_manifest_every > 0 and len(rows) % checkpoint_manifest_every == 0:
            _write_manifest(output_dir, pd.DataFrame(rows))

    videos = pd.DataFrame(rows)
    manifest_path = _write_manifest(output_dir, videos)
    return MacroTranscriptSyncResult(
        videos=videos,
        fetched=fetched,
        skipped=skipped,
        failed=failed,
        transcript_dir=output_dir,
        manifest_path=manifest_path,
    )


def compare_42macro_to_trade_bot(
    *,
    transcript_dir: str | Path,
    warehouse: TradingWarehouse,
    output_dir: str | Path,
    max_match_days: int = 10,
    source: str = DEFAULT_42MACRO_SOURCE,
    include_title_only: bool = True,
) -> MacroAlignmentResult:
    """Classify saved 42 Macro transcripts and compare them to trade-bot history."""

    transcript_root = Path(transcript_dir)
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    videos = _manifest_frame(transcript_root)
    if videos.empty:
        classifications = pd.DataFrame()
        comparisons = pd.DataFrame()
        summary = {"status": "no_transcripts"}
        _write_alignment_outputs(report_dir, videos, classifications, comparisons, summary)
        return MacroAlignmentResult(videos, classifications, comparisons, summary, report_dir)

    classifications = _classify_manifest_transcripts(
        videos,
        transcript_root,
        source=source,
        include_title_only=include_title_only,
    )
    metrics = warehouse.read_table("operating_metric_history")
    comparisons = build_macro_tradebot_comparisons(
        classifications,
        metrics,
        max_match_days=max_match_days,
        source=source,
    )
    summary = summarize_macro_alignment(classifications, comparisons)
    _write_alignment_outputs(report_dir, videos, classifications, comparisons, summary)
    warehouse.save_external_macro_alignment(
        videos=videos,
        classifications=classifications,
        comparisons=comparisons,
    )
    return MacroAlignmentResult(videos, classifications, comparisons, summary, report_dir)


def classify_42macro_transcript(
    text: str,
    *,
    title: str = "",
    published_date: str = "",
    video_id: str = "",
    source: str = DEFAULT_42MACRO_SOURCE,
) -> dict[str, object]:
    combined = f"{title}\n{text}".lower()
    bullish = _weighted_term_score(combined, BULLISH_TERMS)
    defensive = _weighted_term_score(combined, DEFENSIVE_TERMS)
    total = bullish + defensive
    if total <= 0:
        posture_score = 0.0
    else:
        posture_score = (bullish - defensive) / total
    posture_score = float(max(-1.0, min(1.0, posture_score)))
    near_term_risk_score = float(max(0.0, min(1.0, defensive / max(total, 1.0))))
    medium_term_bullish_score = float(max(0.0, min(1.0, bullish / max(total, 1.0))))
    label = _macro_posture_label(posture_score, bullish=bullish, defensive=defensive)
    themes = _themes(combined)
    large_change_flag = any(term in combined for term in LARGE_CHANGE_TERMS)
    return {
        "classification_id": _classification_id(source, video_id or title, published_date),
        "video_id": video_id,
        "source": source,
        "published_date": published_date,
        "title": title,
        "macro_posture_score": posture_score,
        "macro_posture_label": label,
        "near_term_risk_score": near_term_risk_score,
        "medium_term_bullish_score": medium_term_bullish_score,
        "large_change_flag": bool(large_change_flag),
        "bullish_term_score": float(bullish),
        "defensive_term_score": float(defensive),
        "key_themes": ",".join(themes),
        "classification_text_source": "transcript",
        "classification_confidence": 1.0,
        "classified_at_utc": _utc_now_iso(),
    }


def build_macro_tradebot_comparisons(
    classifications: pd.DataFrame,
    operating_metrics: pd.DataFrame,
    *,
    max_match_days: int = 10,
    source: str = DEFAULT_42MACRO_SOURCE,
) -> pd.DataFrame:
    if classifications.empty or operating_metrics.empty:
        return pd.DataFrame()
    metrics = operating_metrics.copy()
    metrics["market_date_dt"] = pd.to_datetime(metrics["market_date"], errors="coerce")
    metrics = metrics.dropna(subset=["market_date_dt"]).sort_values("market_date_dt")
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    previous_trade_score: float | None = None
    previous_risk_score: float | None = None
    for _, row in classifications.sort_values("published_date").iterrows():
        published = pd.to_datetime(row.get("published_date"), errors="coerce")
        if pd.isna(published):
            continue
        matched = _nearest_operating_row(metrics, published, max_match_days=max_match_days)
        if matched is None:
            continue
        trade_posture = _trade_bot_posture_score(matched)
        risk_score = _optional_float(matched.get("risk_score"))
        macro_score = _optional_float(row.get("macro_posture_score")) or 0.0
        disagreement = macro_score - trade_posture
        abs_disagreement = abs(disagreement)
        trade_delta = (
            abs(trade_posture - previous_trade_score)
            if previous_trade_score is not None
            else 0.0
        )
        risk_delta = (
            abs((risk_score or 0.0) - previous_risk_score)
            if previous_risk_score is not None and risk_score is not None
            else 0.0
        )
        large_change_focus = bool(row.get("large_change_flag")) or trade_delta >= 0.20 or risk_delta >= 0.15
        matched_date = pd.Timestamp(matched["market_date_dt"]).date().isoformat()
        rows.append(
            {
                "comparison_id": _comparison_id(source, str(row.get("video_id")), matched_date),
                "video_id": str(row.get("video_id", "")),
                "source": source,
                "published_date": pd.Timestamp(published).date().isoformat(),
                "matched_market_date": matched_date,
                "matched_source": str(matched.get("source", "")),
                "days_from_tradebot": int(
                    abs((pd.Timestamp(published).normalize() - pd.Timestamp(matched["market_date_dt"]).normalize()).days)
                ),
                "macro_posture_score": macro_score,
                "macro_posture_label": str(row.get("macro_posture_label", "")),
                "classification_text_source": str(row.get("classification_text_source", "")),
                "classification_confidence": _optional_float(
                    row.get("classification_confidence")
                ),
                "trade_bot_posture_score": trade_posture,
                "trade_bot_posture_label": _trade_bot_posture_label(trade_posture),
                "disagreement": disagreement,
                "abs_disagreement": abs_disagreement,
                "disagreement_label": _disagreement_label(abs_disagreement),
                "large_change_focus": large_change_focus,
                "trade_bot_risk_score": risk_score,
                "trade_bot_risk_budget_multiplier": _optional_float(
                    matched.get("risk_budget_multiplier")
                ),
                "trade_bot_risk_off_probability": _optional_float(
                    matched.get("one_month_risk_off_probability")
                ),
                "trade_bot_portfolio_risk_multiplier": _optional_float(
                    matched.get("portfolio_risk_multiplier")
                ),
                "notes": _comparison_note(macro_score, trade_posture, large_change_focus),
                "compared_at_utc": _utc_now_iso(),
            }
        )
        previous_trade_score = trade_posture
        previous_risk_score = risk_score
    return pd.DataFrame(rows)


def summarize_macro_alignment(
    classifications: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> dict[str, object]:
    if comparisons.empty:
        return {
            "status": "no_comparisons",
            "transcripts_classified": int(len(classifications)),
            "comparisons": 0,
        }
    macro_mean = float(comparisons["macro_posture_score"].mean())
    trade_mean = float(comparisons["trade_bot_posture_score"].mean())
    major = comparisons[comparisons["disagreement_label"] == "major_mismatch"]
    large = comparisons[comparisons["large_change_focus"].astype(bool)]
    large_major = large[large["disagreement_label"] == "major_mismatch"]
    transcript_backed = comparisons[
        comparisons.get("classification_text_source", pd.Series(dtype=str)).astype(str)
        == "transcript"
    ]
    title_only = comparisons[
        comparisons.get("classification_text_source", pd.Series(dtype=str)).astype(str)
        == "title_only"
    ]
    summary = {
        "status": "ok",
        "transcripts_classified": int(len(classifications)),
        "comparisons": int(len(comparisons)),
        "date_min": str(comparisons["published_date"].min()),
        "date_max": str(comparisons["published_date"].max()),
        "macro_mean_posture_score": macro_mean,
        "trade_bot_mean_posture_score": trade_mean,
        "mean_posture_gap_macro_minus_trade_bot": macro_mean - trade_mean,
        "mean_abs_disagreement": float(comparisons["abs_disagreement"].mean()),
        "major_mismatches": int(len(major)),
        "large_change_comparisons": int(len(large)),
        "large_change_major_mismatches": int(len(large_major)),
        "macro_more_constructive_share": float((comparisons["disagreement"] > 0.35).mean()),
        "trade_bot_more_constructive_share": float((comparisons["disagreement"] < -0.35).mean()),
        "transcript_backed_comparisons": int(len(transcript_backed)),
        "title_only_comparisons": int(len(title_only)),
        "transcript_backed_mean_abs_disagreement": (
            float(transcript_backed["abs_disagreement"].mean())
            if not transcript_backed.empty
            else None
        ),
        "title_only_mean_abs_disagreement": (
            float(title_only["abs_disagreement"].mean()) if not title_only.empty else None
        ),
    }
    top = comparisons.sort_values("abs_disagreement", ascending=False).head(10)
    summary["largest_mismatches"] = top[
        [
            "published_date",
            "video_id",
            "macro_posture_label",
            "trade_bot_posture_label",
            "disagreement",
            "large_change_focus",
            "classification_text_source",
        ]
    ].to_dict(orient="records")
    return summary


def _fetch_channel_video_catalog(
    *,
    session: requests.Session,
    channel_handle: str,
    max_videos: int | None,
    max_pages: int | None,
) -> list[dict[str, object]]:
    html = session.get(
        f"https://www.youtube.com/{channel_handle}/videos",
        timeout=30,
        headers={"User-Agent": _user_agent()},
    ).text
    initial_data = _extract_yt_initial_data(html)
    api_key = _extract_first_regex(html, r'"INNERTUBE_API_KEY":"([^"]+)"')
    client_version = _extract_first_regex(html, r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"')
    visitor_data = _extract_first_regex(html, r'"VISITOR_DATA":"([^"]+)"')
    videos = _extract_lockup_videos(initial_data)
    token = _first_continuation_token(initial_data)
    pages = 1
    while token and (max_pages is None or pages < max_pages):
        if max_videos is not None and len(videos) >= max_videos:
            break
        if not api_key or not client_version:
            break
        payload = {
            "context": {
                "client": {
                    "clientName": "WEB",
                    "clientVersion": client_version,
                    "visitorData": visitor_data,
                }
            },
            "continuation": token,
        }
        response = session.post(
            f"{YOUTUBE_BROWSE_URL}?key={api_key}",
            json=payload,
            timeout=30,
            headers={"User-Agent": _user_agent()},
        )
        response.raise_for_status()
        data = response.json()
        videos.extend(_extract_lockup_videos(data))
        token = _first_continuation_token(data, exclude=token)
        pages += 1
    deduped: dict[str, dict[str, object]] = {}
    for video in videos:
        video_id = str(video.get("video_id", ""))
        if video_id and video_id not in deduped:
            deduped[video_id] = video
    output = list(deduped.values())
    if max_videos is not None:
        return output[:max_videos]
    return output


def _extract_yt_initial_data(html: str) -> dict[str, Any]:
    match = re.search(r"ytInitialData\s*=\s*(\{.*?\});</script>", html, flags=re.S)
    if not match:
        match = re.search(r"var ytInitialData\s*=\s*(\{.*?\});", html, flags=re.S)
    if not match:
        return {}
    return json.loads(match.group(1))


def _extract_lockup_videos(payload: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in _walk_key(payload, "lockupViewModel"):
        if not isinstance(model, dict):
            continue
        video_id = str(model.get("contentId") or _first_nested_value(model, "videoId") or "")
        if not video_id:
            continue
        title = _extract_title(model)
        if title:
            rows.append({"video_id": video_id, "title": title})
    return rows


def _extract_title(model: dict[str, Any]) -> str:
    title = (
        model.get("metadata", {})
        .get("lockupMetadataViewModel", {})
        .get("title", {})
        .get("content")
    )
    if title:
        return str(title)
    for value in _walk_key(model, "title"):
        if isinstance(value, dict) and "content" in value:
            return str(value["content"])
    return ""


def _first_continuation_token(payload: object, *, exclude: str | None = None) -> str | None:
    for command in _walk_key(payload, "continuationCommand"):
        if not isinstance(command, dict):
            continue
        token = command.get("token")
        if isinstance(token, str) and token and token != exclude:
            return token
    return None


def _walk_key(payload: object, key: str) -> list[object]:
    matches: list[object] = []
    if isinstance(payload, dict):
        for item_key, value in payload.items():
            if item_key == key:
                matches.append(value)
            matches.extend(_walk_key(value, key))
    elif isinstance(payload, list):
        for value in payload:
            matches.extend(_walk_key(value, key))
    return matches


def _first_nested_value(payload: object, key: str) -> object | None:
    values = _walk_key(payload, key)
    return values[0] if values else None


def _fetch_video_publish_date(session: requests.Session, video_id: str) -> str | None:
    response = session.get(
        _youtube_url(video_id),
        timeout=30,
        headers={"User-Agent": _user_agent()},
    )
    response.raise_for_status()
    html = response.text
    for pattern in (
        r'"uploadDate":"([^"]+)"',
        r'<meta itemprop="datePublished" content="([^"]+)"',
    ):
        value = _extract_first_regex(html, pattern)
        if value:
            return value[:10]
    return None


def _fetch_youtube_transcript(video_id: str, *, timeout_seconds: int = 30) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "youtube-transcript-api is required. Run `poetry install --sync`."
        ) from exc

    with _timeout(timeout_seconds):
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=("en",))
    snippets = [str(snippet.text).strip() for snippet in transcript if str(snippet.text).strip()]
    return "\n".join(snippets)


class _TranscriptTimeout(RuntimeError):
    pass


class _timeout:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._previous_handler: Any = None

    def __enter__(self) -> None:
        if self.seconds <= 0:
            return
        self._previous_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.seconds <= 0:
            return
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._previous_handler)

    def _handle_timeout(self, signum: int, frame: object) -> None:
        raise _TranscriptTimeout(f"Transcript fetch timed out after {self.seconds} seconds.")


def _load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _manifest_frame(transcript_root: Path) -> pd.DataFrame:
    manifest_path = transcript_root / "manifest.json"
    rows = _load_manifest(manifest_path)
    if rows:
        normalized_rows = []
        for row in rows:
            transcript_path = str(row.get("transcript_path") or row.get("path") or "")
            resolved = _resolve_transcript_path(transcript_root, transcript_path)
            normalized_rows.append(
                {
                    "video_id": str(row.get("video_id", "")),
                    "source": str(row.get("source", DEFAULT_42MACRO_SOURCE)),
                    "published_date": str(row.get("published_date") or row.get("date") or ""),
                    "title": str(row.get("title", "")),
                    "url": str(row.get("url", "")),
                    "transcript_path": (
                        str(resolved.relative_to(transcript_root))
                        if resolved is not None
                        and resolved.exists()
                        and resolved.is_relative_to(transcript_root)
                        else transcript_path
                    ),
                    "word_count": int(row.get("word_count") or row.get("words") or 0),
                    "fetched_at_utc": str(row.get("fetched_at_utc", "")),
                    "status": str(row.get("status", "")),
                    "error": str(row.get("error", "")),
                }
            )
        frame = pd.DataFrame(normalized_rows)
        local = pd.DataFrame(_local_transcript_rows(transcript_root))
        if local.empty:
            return frame
        known_ids = set(frame["video_id"].astype(str)) if "video_id" in frame else set()
        additions = local[~local["video_id"].astype(str).isin(known_ids)]
        if additions.empty:
            return frame
        return pd.concat([frame, additions], ignore_index=True)
    return pd.DataFrame(_local_transcript_rows(transcript_root))


def _local_transcript_rows(transcript_root: Path) -> list[dict[str, object]]:
    fallback_rows = []
    for path in sorted(transcript_root.glob("*.txt")):
        header = _parse_transcript_header(path)
        published_date = header.get("date") or header.get("published_date", "")
        video_id = header.get("video_id", "") or _video_id_from_filename(path)
        fallback_rows.append(
            {
                "video_id": video_id,
                "source": header.get("source", DEFAULT_42MACRO_SOURCE),
                "published_date": published_date,
                "title": header.get("title", path.stem),
                "url": header.get("url", ""),
                "transcript_path": path.name,
                "word_count": _word_count(path.read_text(encoding="utf-8", errors="ignore")),
                "fetched_at_utc": "",
                "status": "local_file",
                "error": "",
            }
        )
    return fallback_rows


def _parse_transcript_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:10]:
        if not line.strip():
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return header


def _video_id_from_filename(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[1] if len(parts) >= 2 else ""


def _resolve_transcript_path(transcript_root: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    rooted = transcript_root / path
    if rooted.exists():
        return rooted
    if path.exists():
        return path
    return rooted


def _relative_transcript_path(transcript_root: Path, path: Path) -> str:
    if not path or str(path) in {"", "."} or not path.exists() or path.is_dir():
        return ""
    try:
        return str(path.relative_to(transcript_root))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(transcript_root.resolve()))
    except ValueError:
        return str(path)


def _classify_manifest_transcripts(
    videos: pd.DataFrame,
    transcript_root: Path,
    *,
    source: str,
    include_title_only: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, video in videos.iterrows():
        transcript_path = _resolve_transcript_path(
            transcript_root,
            str(video.get("transcript_path", "")),
        )
        if transcript_path is not None and transcript_path.exists() and not transcript_path.is_dir():
            text = transcript_path.read_text(encoding="utf-8", errors="ignore")
            classification = classify_42macro_transcript(
                _strip_transcript_header(text),
                title=str(video.get("title", "")),
                published_date=str(video.get("published_date", "")),
                video_id=str(video.get("video_id", "")),
                source=source,
            )
            rows.append(classification)
        elif include_title_only:
            classification = classify_42macro_transcript(
                "",
                title=str(video.get("title", "")),
                published_date=str(video.get("published_date", "")),
                video_id=str(video.get("video_id", "")),
                source=source,
            )
            classification["classification_text_source"] = "title_only"
            classification["classification_confidence"] = 0.25
            rows.append(classification)
    return pd.DataFrame(rows)


def _strip_transcript_header(text: str) -> str:
    parts = text.split("\n\n", 1)
    if len(parts) == 2 and "video_id:" in parts[0]:
        return parts[1]
    return text


def _nearest_operating_row(
    metrics: pd.DataFrame,
    date_value: pd.Timestamp,
    *,
    max_match_days: int,
) -> pd.Series | None:
    target = pd.Timestamp(date_value).normalize()
    deltas = (metrics["market_date_dt"].dt.normalize() - target).abs()
    index = deltas.idxmin()
    days = int(deltas.loc[index].days)
    if days > max_match_days:
        return None
    return metrics.loc[index]


def _trade_bot_posture_score(row: pd.Series) -> float:
    budget = _optional_float(row.get("risk_budget_multiplier"))
    risk_score = _optional_float(row.get("risk_score"))
    risk_off = _optional_float(row.get("one_month_risk_off_probability"))
    if budget is None:
        budget = 1.0 - (risk_score or 0.5)
    budget_component = (budget * 2.0) - 1.0
    risk_component = (0.5 - (risk_score or 0.5)) * 1.2
    risk_off_component = (0.30 - (risk_off or 0.30)) * 0.8
    return float(max(-1.0, min(1.0, budget_component + risk_component + risk_off_component)))


def _trade_bot_posture_label(score: float) -> str:
    if score >= 0.45:
        return "risk_on"
    if score >= 0.15:
        return "constructive"
    if score <= -0.45:
        return "defensive"
    if score <= -0.15:
        return "cautious"
    return "balanced"


def _macro_posture_label(score: float, *, bullish: float, defensive: float) -> str:
    if bullish > 0 and defensive > 0 and abs(score) < 0.30:
        return "constructive_but_fragile"
    if score >= 0.45:
        return "risk_on"
    if score >= 0.15:
        return "constructive"
    if score <= -0.45:
        return "risk_reduction"
    if score <= -0.15:
        return "cautious"
    return "balanced"


def _disagreement_label(abs_disagreement: float) -> str:
    if abs_disagreement >= 0.70:
        return "major_mismatch"
    if abs_disagreement >= 0.35:
        return "modest_mismatch"
    return "aligned"


def _comparison_note(macro_score: float, trade_score: float, large_change_focus: bool) -> str:
    if macro_score - trade_score > 0.35:
        direction = "42 Macro is more constructive than trade-bot."
    elif trade_score - macro_score > 0.35:
        direction = "Trade-bot is more constructive than 42 Macro."
    else:
        direction = "Postures are broadly aligned."
    if large_change_focus:
        return f"{direction} Large-change context should be reviewed."
    return direction


def _weighted_term_score(text: str, weights: dict[str, float]) -> float:
    score = 0.0
    for term, weight in weights.items():
        score += text.count(term) * weight
    return score


def _themes(text: str) -> list[str]:
    themes = []
    for theme, terms in THEME_TERMS.items():
        if any(term in text for term in terms):
            themes.append(theme)
    return themes


def _write_alignment_outputs(
    output_dir: Path,
    videos: pd.DataFrame,
    classifications: pd.DataFrame,
    comparisons: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    videos.to_csv(output_dir / "videos.csv", index=False)
    classifications.to_csv(output_dir / "classified_transcripts.csv", index=False)
    comparisons.to_csv(output_dir / "daily_comparison.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        _summary_markdown(summary, comparisons),
        encoding="utf-8",
    )


def _write_manifest(output_dir: Path, videos: pd.DataFrame) -> Path:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(videos.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    videos.to_csv(output_dir / "manifest.csv", index=False)
    return manifest_path


def _summary_markdown(summary: dict[str, object], comparisons: pd.DataFrame) -> str:
    if summary.get("status") != "ok":
        return f"# 42 Macro / Trade-Bot Alignment\n\nStatus: {summary.get('status')}\n"
    lines = [
        "# 42 Macro / Trade-Bot Alignment",
        "",
        f"Compared videos: {summary['comparisons']}",
        f"Date range: {summary['date_min']} to {summary['date_max']}",
        f"Mean 42 Macro posture score: {summary['macro_mean_posture_score']:.2f}",
        f"Mean trade-bot posture score: {summary['trade_bot_mean_posture_score']:.2f}",
        f"Mean absolute disagreement: {summary['mean_abs_disagreement']:.2f}",
        f"Major mismatches: {summary['major_mismatches']}",
        f"Large-change comparisons: {summary['large_change_comparisons']}",
        f"Large-change major mismatches: {summary['large_change_major_mismatches']}",
        "",
        "## Largest Mismatches",
        "",
    ]
    if comparisons.empty:
        lines.append("No comparisons were produced.")
        return "\n".join(lines) + "\n"
    top = comparisons.sort_values("abs_disagreement", ascending=False).head(10)
    lines.append(
        "| date | video_id | 42 Macro | trade-bot | gap | large-change | note |"
    )
    lines.append("|---|---|---|---|---:|---|---|")
    for _, row in top.iterrows():
        lines.append(
            "| "
            f"{row['published_date']} | {row['video_id']} | {row['macro_posture_label']} | "
            f"{row['trade_bot_posture_label']} | {row['disagreement']:.2f} | "
            f"{bool(row['large_change_focus'])} | {row['notes']} |"
        )
    return "\n".join(lines) + "\n"


def _format_transcript_file(
    *,
    source: str,
    published_date: str,
    video_id: str,
    title: str,
    url: str,
    transcript: str,
) -> str:
    header = "\n".join(
        [
            f"source: {source}",
            f"published_date: {published_date}",
            f"video_id: {video_id}",
            f"url: {url}",
            f"title: {title}",
        ]
    )
    return f"{header}\n\n{transcript.strip()}\n"


def _transcript_filename(published_date: str, video_id: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80] or "video"
    date_label = published_date if published_date else "unknown-date"
    return f"{date_label}_{video_id}_{slug}.txt"


def _classification_id(source: str, video_id: str, published_date: str) -> str:
    return f"{source}:{published_date}:{video_id}"


def _comparison_id(source: str, video_id: str, matched_date: str) -> str:
    return f"{source}:{video_id}:{matched_date}"


def _youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _extract_first_regex(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _user_agent() -> str:
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
