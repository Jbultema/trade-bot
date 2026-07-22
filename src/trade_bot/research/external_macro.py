from __future__ import annotations

import json
import re
import signal
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from trade_bot.storage.warehouse import TradingWarehouse

YOUTUBE_BROWSE_URL = "https://www.youtube.com/youtubei/v1/browse"
DEFAULT_42MACRO_HANDLE = "@42Macro"
DEFAULT_42MACRO_SOURCE = "42macro_youtube"
DEFAULT_OUTCOME_HORIZONS = {"1w": 5, "1m": 21, "3m": 63}
DEFAULT_OUTCOME_TICKERS = ("SPY", "QQQ", "GLD", "VEA", "IWM")


@dataclass(frozen=True)
class MacroTranscriptSyncResult:
    videos: pd.DataFrame
    fetched: int
    skipped: int
    failed: int
    transcript_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class MacroTranscriptImportResult:
    videos: pd.DataFrame
    imported: int
    skipped: int
    failed: int
    transcript_dir: Path
    manifest_path: Path
    imported_files: pd.DataFrame


@dataclass(frozen=True)
class MacroAlignmentResult:
    videos: pd.DataFrame
    classifications: pd.DataFrame
    comparisons: pd.DataFrame
    summary: dict[str, object]
    output_dir: Path


@dataclass(frozen=True)
class MacroOutcomeResult:
    outcomes: pd.DataFrame
    summary: pd.DataFrame
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
    local_transcripts = _local_transcript_rows(output_dir)
    prior_by_id = {str(row.get("video_id")): row for row in prior_manifest}
    for local_row in local_transcripts:
        video_id = str(local_row.get("video_id", ""))
        if video_id and not str(prior_by_id.get(video_id, {}).get("transcript_path", "")):
            prior_by_id[video_id] = local_row
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
        if existing_path is not None and existing_path.exists() and not refresh and publish_date:
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


def import_42macro_transcript_files(
    *,
    input_dir: str | Path,
    transcript_dir: str | Path,
    source: str = DEFAULT_42MACRO_SOURCE,
    overwrite: bool = False,
) -> MacroTranscriptImportResult:
    """Import browser-copied transcript text or saved HTML transcript pages.

    This is the fallback path for public transcript pages that render in a
    normal browser but block direct non-browser HTTP clients.
    """

    import_root = Path(input_dir)
    output_dir = Path(transcript_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_frame(output_dir)
    manifest_by_id = {
        str(row.get("video_id", "")): dict(row)
        for _, row in manifest.iterrows()
        if str(row.get("video_id", ""))
    }
    manifest_by_title = {
        _normalize_title(str(row.get("title", ""))): dict(row)
        for _, row in manifest.iterrows()
        if str(row.get("title", ""))
    }
    imported_rows: list[dict[str, object]] = []
    imported = 0
    skipped = 0
    failed = 0
    supported = {".txt", ".md", ".html", ".htm"}
    for input_path in sorted(
        path for path in import_root.rglob("*") if path.suffix.lower() in supported
    ):
        try:
            raw = input_path.read_text(encoding="utf-8", errors="ignore")
            header = _parse_transcript_header_text(raw)
            title = (
                header.get("title", "")
                or _extract_youtube_transcript_page_title(raw)
                or input_path.stem
            )
            video_id = (
                header.get("video_id", "")
                or _extract_video_id(f"{input_path.name}\n{raw}")
                or _video_id_for_title(title, manifest_by_title)
            )
            if not video_id:
                raise ValueError(
                    "Could not infer YouTube video ID from file name, header, URL, or title."
                )
            manifest_row = manifest_by_id.get(video_id, {})
            title = str(manifest_row.get("title") or title)
            published_date = (
                header.get("published_date", "")
                or header.get("date", "")
                or str(manifest_row.get("published_date", ""))
                or _date_from_text_or_filename(f"{input_path.name}\n{raw}")
            )
            url = (
                header.get("url", "") or str(manifest_row.get("url", "")) or _youtube_url(video_id)
            )
            transcript = _extract_manual_transcript_text(raw)
            if _word_count(transcript) < 20:
                raise ValueError("Transcript text is too short after removing page chrome.")
            output_path = output_dir / _transcript_filename(published_date, video_id, title)
            if output_path.exists() and not overwrite:
                skipped += 1
                status = "skipped_existing"
            else:
                output_path.write_text(
                    _format_transcript_file(
                        source=source,
                        published_date=published_date,
                        video_id=video_id,
                        title=title,
                        url=url,
                        transcript=transcript,
                    ),
                    encoding="utf-8",
                )
                imported += 1
                status = "imported_manual"
            row = {
                "video_id": video_id,
                "source": source,
                "published_date": published_date,
                "title": title,
                "url": url,
                "transcript_path": _relative_transcript_path(output_dir, output_path),
                "word_count": _word_count(transcript),
                "fetched_at_utc": _utc_now_iso(),
                "status": status,
                "error": "",
                "input_path": str(input_path),
            }
            imported_rows.append(row)
            manifest_by_id[video_id] = {
                key: value for key, value in row.items() if key != "input_path"
            }
        except Exception as exc:
            failed += 1
            imported_rows.append(
                {
                    "input_path": str(input_path),
                    "video_id": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )
    videos = pd.DataFrame(manifest_by_id.values())
    videos = _sort_manifest_frame(videos)
    manifest_path = _write_manifest(output_dir, videos)
    return MacroTranscriptImportResult(
        videos=videos,
        imported=imported,
        skipped=skipped,
        failed=failed,
        transcript_dir=output_dir,
        manifest_path=manifest_path,
        imported_files=pd.DataFrame(imported_rows),
    )


def write_missing_42macro_transcript_priority(
    *,
    transcript_dir: str | Path,
    output_dir: str | Path,
    comparison_path: str | Path | None = None,
    outcome_path: str | Path | None = None,
) -> pd.DataFrame:
    """Write a ranked queue of missing transcript URLs worth filling first."""

    transcript_root = Path(transcript_dir)
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    videos = _manifest_frame(transcript_root)
    if videos.empty:
        priority = pd.DataFrame()
        priority.to_csv(report_dir / "missing_transcript_priority.csv", index=False)
        return priority
    videos = videos.copy()
    videos["has_transcript"] = videos["transcript_path"].apply(
        lambda value: (
            _resolve_transcript_path(transcript_root, str(value)) is not None
            and _resolve_transcript_path(transcript_root, str(value)).exists()
        )
    )
    missing = videos[~videos["has_transcript"]].copy()
    if missing.empty:
        priority = pd.DataFrame(
            columns=[
                "priority_score",
                "published_date",
                "video_id",
                "title",
                "url",
                "transcript_url",
                "priority_reason",
            ]
        )
        priority.to_csv(report_dir / "missing_transcript_priority.csv", index=False)
        return priority

    comparison_file = (
        Path(comparison_path) if comparison_path else report_dir / "daily_comparison.csv"
    )
    outcome_file = Path(outcome_path) if outcome_path else report_dir / "forward_outcome_scores.csv"
    comparisons = pd.read_csv(comparison_file) if comparison_file.exists() else pd.DataFrame()
    outcomes = pd.read_csv(outcome_file) if outcome_file.exists() else pd.DataFrame()
    comparison_by_id = (
        {str(video_id): frame for video_id, frame in comparisons.groupby("video_id", dropna=False)}
        if not comparisons.empty and "video_id" in comparisons
        else {}
    )
    outcome_by_id = (
        {str(video_id): frame for video_id, frame in outcomes.groupby("video_id", dropna=False)}
        if not outcomes.empty and "video_id" in outcomes
        else {}
    )

    rows: list[dict[str, object]] = []
    newest_date = pd.to_datetime(missing["published_date"], errors="coerce").max()
    for _, video in missing.iterrows():
        video_id = str(video.get("video_id", ""))
        score = 0.0
        reasons: list[str] = []
        comparison_frame = comparison_by_id.get(video_id, pd.DataFrame())
        if not comparison_frame.empty:
            if bool(
                comparison_frame.get("large_change_focus", pd.Series(dtype=bool)).astype(bool).any()
            ):
                score += 100.0
                reasons.append("large-change comparison")
            max_gap = float(comparison_frame.get("abs_disagreement", pd.Series([0.0])).max())
            if max_gap >= 0.70:
                score += 40.0
                reasons.append("major 42/trade-bot mismatch")
            elif max_gap >= 0.35:
                score += 20.0
                reasons.append("modest 42/trade-bot mismatch")
        outcome_frame = outcome_by_id.get(video_id, pd.DataFrame())
        if not outcome_frame.empty:
            large_outcomes = outcome_frame[
                outcome_frame.get("realized_environment", pd.Series(dtype=str))
                .astype(str)
                .isin(["left_tail", "constructive"])
            ]
            if not large_outcomes.empty:
                score += 60.0
                reasons.append("large realized forward move")
            if {"macro_action_score", "trade_bot_action_score"}.issubset(outcome_frame.columns):
                action_gap = (
                    (
                        outcome_frame["macro_action_score"].astype(float)
                        - outcome_frame["trade_bot_action_score"].astype(float)
                    )
                    .abs()
                    .max()
                )
                if pd.notna(action_gap) and float(action_gap) >= 0.35:
                    score += 30.0
                    reasons.append("large outcome-score gap")
        published = pd.to_datetime(video.get("published_date"), errors="coerce")
        if pd.notna(published) and pd.notna(newest_date):
            age_days = max(0, int((newest_date.normalize() - published.normalize()).days))
            recency_score = max(0.0, 30.0 - min(age_days, 365) / 365.0 * 30.0)
            score += recency_score
            if age_days <= 45:
                reasons.append("recent")
        if not reasons:
            reasons.append("missing transcript")
        rows.append(
            {
                "priority_score": round(score, 3),
                "published_date": str(video.get("published_date", "")),
                "video_id": video_id,
                "title": str(video.get("title", "")),
                "url": str(video.get("url", "")) or _youtube_url(video_id),
                "transcript_url": f"https://youtubetotranscript.com/transcript?v={video_id}",
                "priority_reason": "; ".join(dict.fromkeys(reasons)),
            }
        )
    priority = pd.DataFrame(rows).sort_values(
        ["priority_score", "published_date"],
        ascending=[False, False],
    )
    priority.to_csv(report_dir / "missing_transcript_priority.csv", index=False)
    return priority


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


def score_macro_tradebot_outcomes(
    *,
    comparisons: pd.DataFrame,
    prices: pd.DataFrame,
    output_dir: str | Path,
    horizons: dict[str, int] | None = None,
    risk_ticker: str = "SPY",
    defensive_ticker: str = "BIL",
    context_tickers: tuple[str, ...] = DEFAULT_OUTCOME_TICKERS,
) -> MacroOutcomeResult:
    """Score 42 Macro and trade-bot posture against forward market outcomes."""

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    outcome_horizons = horizons or DEFAULT_OUTCOME_HORIZONS
    outcomes = build_forward_outcome_scores(
        comparisons,
        prices,
        horizons=outcome_horizons,
        risk_ticker=risk_ticker,
        defensive_ticker=defensive_ticker,
        context_tickers=context_tickers,
    )
    summary = summarize_forward_outcome_scores(outcomes)
    outcomes.to_csv(report_dir / "forward_outcome_scores.csv", index=False)
    summary.to_csv(report_dir / "forward_outcome_summary.csv", index=False)
    (report_dir / "outcome_analysis.md").write_text(
        _outcome_summary_markdown(outcomes, summary),
        encoding="utf-8",
    )
    return MacroOutcomeResult(outcomes=outcomes, summary=summary, output_dir=report_dir)


def build_forward_outcome_scores(
    comparisons: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: dict[str, int] | None = None,
    risk_ticker: str = "SPY",
    defensive_ticker: str = "BIL",
    context_tickers: tuple[str, ...] = DEFAULT_OUTCOME_TICKERS,
) -> pd.DataFrame:
    if comparisons.empty or prices.empty:
        return pd.DataFrame()
    outcome_horizons = horizons or DEFAULT_OUTCOME_HORIZONS
    clean_prices = prices.sort_index().ffill()
    clean_prices.index = pd.to_datetime(clean_prices.index, errors="coerce")
    clean_prices = clean_prices[~clean_prices.index.isna()].sort_index()
    if risk_ticker not in clean_prices:
        return pd.DataFrame()
    if defensive_ticker not in clean_prices:
        clean_prices[defensive_ticker] = 1.0

    rows: list[dict[str, object]] = []
    for _, comparison in comparisons.iterrows():
        origin_date = pd.to_datetime(
            comparison.get("matched_market_date") or comparison.get("published_date"),
            errors="coerce",
        )
        if pd.isna(origin_date):
            continue
        origin_index = _first_market_index_on_or_after(
            clean_prices.index, pd.Timestamp(origin_date)
        )
        if origin_index is None:
            continue
        for horizon_label, horizon_days in outcome_horizons.items():
            end_index = origin_index + int(horizon_days)
            if end_index >= len(clean_prices.index):
                continue
            start_date = clean_prices.index[origin_index]
            end_date = clean_prices.index[end_index]
            risk_return = _forward_return(clean_prices, risk_ticker, origin_index, end_index)
            defensive_return = _forward_return(
                clean_prices,
                defensive_ticker,
                origin_index,
                end_index,
            )
            if risk_return is None or defensive_return is None:
                continue
            risk_drawdown = _forward_max_drawdown(
                clean_prices,
                risk_ticker,
                origin_index,
                end_index,
            )
            context = _context_forward_returns(
                clean_prices,
                context_tickers,
                origin_index,
                end_index,
            )
            excess_return = risk_return - defensive_return
            left_tail_threshold = _left_tail_threshold(horizon_label)
            constructive_threshold = _constructive_threshold(horizon_label)
            environment = _realized_environment(
                excess_return=excess_return,
                risk_return=risk_return,
                risk_drawdown=risk_drawdown,
                left_tail_threshold=left_tail_threshold,
                constructive_threshold=constructive_threshold,
            )
            macro_score = _optional_float(comparison.get("macro_posture_score")) or 0.0
            trade_score = _optional_float(comparison.get("trade_bot_posture_score")) or 0.0
            macro_allocation = _posture_to_allocation(macro_score)
            trade_allocation = _posture_to_allocation(trade_score)
            macro_proxy_return = _proxy_return(macro_allocation, risk_return, defensive_return)
            trade_proxy_return = _proxy_return(trade_allocation, risk_return, defensive_return)
            oracle_proxy_return = max(risk_return, defensive_return)
            macro_action_score = _risk_sizing_action_score(macro_allocation, environment)
            trade_action_score = _risk_sizing_action_score(trade_allocation, environment)
            row = {
                "video_id": str(comparison.get("video_id", "")),
                "published_date": str(comparison.get("published_date", "")),
                "origin_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "horizon": horizon_label,
                "horizon_days": int(horizon_days),
                "classification_text_source": str(comparison.get("classification_text_source", "")),
                "classification_confidence": _optional_float(
                    comparison.get("classification_confidence")
                ),
                "macro_posture_score": macro_score,
                "trade_bot_posture_score": trade_score,
                "macro_risk_allocation_proxy": macro_allocation,
                "trade_bot_risk_allocation_proxy": trade_allocation,
                "risk_ticker": risk_ticker,
                "risk_forward_return": risk_return,
                "defensive_forward_return": defensive_return,
                "risk_excess_return": excess_return,
                "risk_forward_max_drawdown": risk_drawdown,
                "realized_environment": environment,
                "macro_proxy_return": macro_proxy_return,
                "trade_bot_proxy_return": trade_proxy_return,
                "oracle_proxy_return": oracle_proxy_return,
                "macro_return_regret": oracle_proxy_return - macro_proxy_return,
                "trade_bot_return_regret": oracle_proxy_return - trade_proxy_return,
                "macro_action_score": macro_action_score,
                "trade_bot_action_score": trade_action_score,
                "macro_correct_side": _correct_side(macro_score, excess_return),
                "trade_bot_correct_side": _correct_side(trade_score, excess_return),
                "macro_overrisk": environment == "left_tail" and macro_allocation > 0.60,
                "trade_bot_overrisk": environment == "left_tail" and trade_allocation > 0.60,
                "macro_underrisk": environment == "constructive" and macro_allocation < 0.40,
                "trade_bot_underrisk": environment == "constructive" and trade_allocation < 0.40,
                "proxy_return_winner": _winner_label(
                    macro_proxy_return,
                    trade_proxy_return,
                    "42macro",
                    "trade_bot",
                ),
                "action_score_winner": _winner_label(
                    macro_action_score,
                    trade_action_score,
                    "42macro",
                    "trade_bot",
                ),
            }
            row.update(context)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_forward_outcome_scores(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ["classification_text_source", "horizon"]
    for keys, frame in outcomes.groupby(group_columns, dropna=False):
        text_source, horizon = keys
        rows.append(_outcome_summary_row(str(text_source), str(horizon), frame))
    rows.append(_outcome_summary_row("all", "all", outcomes))
    for horizon, frame in outcomes.groupby("horizon", dropna=False):
        rows.append(_outcome_summary_row("all", str(horizon), frame))
    return pd.DataFrame(rows)


def _outcome_summary_row(scope: str, horizon: str, frame: pd.DataFrame) -> dict[str, object]:
    left_tail = frame[frame["realized_environment"] == "left_tail"]
    constructive = frame[frame["realized_environment"] == "constructive"]
    return {
        "scope": scope,
        "horizon": horizon,
        "rows": int(len(frame)),
        "date_min": str(frame["origin_date"].min()),
        "date_max": str(frame["origin_date"].max()),
        "risk_mean_forward_return": float(frame["risk_forward_return"].mean()),
        "risk_mean_excess_return": float(frame["risk_excess_return"].mean()),
        "left_tail_rows": int(len(left_tail)),
        "constructive_rows": int(len(constructive)),
        "macro_mean_proxy_return": float(frame["macro_proxy_return"].mean()),
        "trade_bot_mean_proxy_return": float(frame["trade_bot_proxy_return"].mean()),
        "macro_mean_return_regret": float(frame["macro_return_regret"].mean()),
        "trade_bot_mean_return_regret": float(frame["trade_bot_return_regret"].mean()),
        "macro_mean_action_score": float(frame["macro_action_score"].mean()),
        "trade_bot_mean_action_score": float(frame["trade_bot_action_score"].mean()),
        "macro_correct_side_rate": float(frame["macro_correct_side"].mean()),
        "trade_bot_correct_side_rate": float(frame["trade_bot_correct_side"].mean()),
        "macro_overrisk_rate": float(frame["macro_overrisk"].mean()),
        "trade_bot_overrisk_rate": float(frame["trade_bot_overrisk"].mean()),
        "macro_underrisk_rate": float(frame["macro_underrisk"].mean()),
        "trade_bot_underrisk_rate": float(frame["trade_bot_underrisk"].mean()),
        "macro_left_tail_overrisk_rate": (
            float(left_tail["macro_overrisk"].mean()) if not left_tail.empty else np.nan
        ),
        "trade_bot_left_tail_overrisk_rate": (
            float(left_tail["trade_bot_overrisk"].mean()) if not left_tail.empty else np.nan
        ),
        "macro_constructive_underrisk_rate": (
            float(constructive["macro_underrisk"].mean()) if not constructive.empty else np.nan
        ),
        "trade_bot_constructive_underrisk_rate": (
            float(constructive["trade_bot_underrisk"].mean()) if not constructive.empty else np.nan
        ),
    }


def _first_market_index_on_or_after(
    index: pd.DatetimeIndex, date_value: pd.Timestamp
) -> int | None:
    positions = np.flatnonzero(index.normalize() >= date_value.normalize())
    if len(positions) == 0:
        return None
    return int(positions[0])


def _forward_return(
    prices: pd.DataFrame,
    ticker: str,
    origin_index: int,
    end_index: int,
) -> float | None:
    if ticker not in prices:
        return None
    series = prices[ticker].astype(float).iloc[[origin_index, end_index]]
    if series.isna().any() or float(series.iloc[0]) == 0.0:
        return None
    return float(series.iloc[1] / series.iloc[0] - 1.0)


def _forward_max_drawdown(
    prices: pd.DataFrame,
    ticker: str,
    origin_index: int,
    end_index: int,
) -> float:
    series = prices[ticker].astype(float).iloc[origin_index : end_index + 1].dropna()
    if series.empty or float(series.iloc[0]) == 0.0:
        return float("nan")
    relative = series / float(series.iloc[0])
    drawdown = relative / relative.cummax() - 1.0
    return float(drawdown.min())


def _context_forward_returns(
    prices: pd.DataFrame,
    tickers: tuple[str, ...],
    origin_index: int,
    end_index: int,
) -> dict[str, float]:
    rows: dict[str, float] = {}
    for ticker in tickers:
        value = _forward_return(prices, ticker, origin_index, end_index)
        if value is not None:
            rows[f"{ticker.lower()}_forward_return"] = value
    return rows


def _left_tail_threshold(horizon: str) -> float:
    return {"1w": -0.025, "1m": -0.055, "3m": -0.085}.get(horizon, -0.055)


def _constructive_threshold(horizon: str) -> float:
    return {"1w": 0.010, "1m": 0.025, "3m": 0.050}.get(horizon, 0.025)


def _realized_environment(
    *,
    excess_return: float,
    risk_return: float,
    risk_drawdown: float,
    left_tail_threshold: float,
    constructive_threshold: float,
) -> str:
    if risk_return <= left_tail_threshold or risk_drawdown <= left_tail_threshold:
        return "left_tail"
    if excess_return >= constructive_threshold and risk_drawdown > left_tail_threshold / 2.0:
        return "constructive"
    return "choppy"


def _posture_to_allocation(score: float) -> float:
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def _proxy_return(allocation: float, risk_return: float, defensive_return: float) -> float:
    return float(allocation * risk_return + (1.0 - allocation) * defensive_return)


def _risk_sizing_action_score(allocation: float, environment: str) -> float:
    if environment == "constructive":
        return float(allocation)
    if environment == "left_tail":
        return float(1.0 - allocation)
    return float(1.0 - abs(allocation - 0.50) * 2.0)


def _correct_side(score: float, excess_return: float) -> bool:
    if excess_return > 0:
        return score >= 0
    if excess_return < 0:
        return score <= 0
    return abs(score) < 0.15


def _winner_label(left: float, right: float, left_label: str, right_label: str) -> str:
    if abs(left - right) < 1e-9:
        return "tie"
    return left_label if left > right else right_label


def _outcome_summary_markdown(outcomes: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# 42 Macro / Trade-Bot Forward Outcome Analysis",
        "",
        "## Goal",
        "",
        (
            "Measure whether 42 Macro's public tactical read and trade-bot's "
            "risk posture were appropriately sized for what happened next."
        ),
        "",
        "## Method",
        "",
        "- Convert each system's daily posture into a 0-100% risk-allocation proxy.",
        "- Score forward 1w, 1m, and 3m SPY-versus-BIL outcomes from the matched date.",
        "- Reward risk-on posture in constructive forward windows.",
        "- Reward defensive posture before left-tail return/drawdown windows.",
        "- Keep transcript-backed and title-only 42 Macro classifications separate.",
        "",
    ]
    if outcomes.empty or summary.empty:
        lines.append("No forward outcome rows were produced.")
        return "\n".join(lines) + "\n"

    transcript = summary[summary["scope"].eq("transcript")]
    if not transcript.empty:
        lines.extend(["## Transcript-Backed Summary", ""])
        for _, row in transcript.sort_values("horizon").iterrows():
            lines.append(
                f"- {row['horizon']}: {int(row['rows']):,} rows; "
                f"42 Macro action score {row['macro_mean_action_score']:.2f}, "
                f"trade-bot action score {row['trade_bot_mean_action_score']:.2f}; "
                f"42 Macro proxy return {row['macro_mean_proxy_return']:.2%}, "
                f"trade-bot proxy return {row['trade_bot_mean_proxy_return']:.2%}."
            )
        lines.append("")

    all_horizons = summary[summary["scope"].eq("all") & ~summary["horizon"].eq("all")]
    if not all_horizons.empty:
        lines.extend(["## Whole-Corpus Directional Summary", ""])
        for _, row in all_horizons.sort_values("horizon").iterrows():
            lines.append(
                f"- {row['horizon']}: {int(row['rows']):,} rows; "
                f"42 Macro action score {row['macro_mean_action_score']:.2f}, "
                f"trade-bot action score {row['trade_bot_mean_action_score']:.2f}; "
                f"left-tail rows {int(row['left_tail_rows']):,}, "
                f"constructive rows {int(row['constructive_rows']):,}."
            )
        lines.append("")

    large = outcomes[
        (outcomes["classification_text_source"].eq("transcript"))
        & (outcomes["realized_environment"].isin(["left_tail", "constructive"]))
    ].copy()
    if not large.empty:
        large["action_score_gap"] = large["macro_action_score"] - large["trade_bot_action_score"]
        top = large.reindex(
            large["action_score_gap"].abs().sort_values(ascending=False).index
        ).head(12)
        lines.extend(["## Largest Transcript-Backed Outcome Gaps", ""])
        lines.append(
            "| date | horizon | environment | 42 score | bot score | gap | SPY fwd | SPY max DD |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in top.iterrows():
            lines.append(
                f"| {row['origin_date']} | {row['horizon']} | {row['realized_environment']} | "
                f"{row['macro_action_score']:.2f} | {row['trade_bot_action_score']:.2f} | "
                f"{row['action_score_gap']:.2f} | {row['risk_forward_return']:.2%} | "
                f"{row['risk_forward_max_drawdown']:.2%} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Caveats",
            "",
            "- This is a tactical sizing audit, not a full test of 42 Macro's long-horizon themes.",
            "- A transcript-derived heuristic is still not the same as 42 Macro's proprietary model state.",
            "- Title-only rows are included only for coverage and prioritization; conclusions should emphasize transcript-backed rows.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    posture_score = 0.0 if total <= 0 else (bullish - defensive) / total
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
            abs(trade_posture - previous_trade_score) if previous_trade_score is not None else 0.0
        )
        risk_delta = (
            abs((risk_score or 0.0) - previous_risk_score)
            if previous_risk_score is not None and risk_score is not None
            else 0.0
        )
        large_change_focus = (
            bool(row.get("large_change_flag")) or trade_delta >= 0.20 or risk_delta >= 0.15
        )
        matched_date = pd.Timestamp(matched["market_date_dt"]).date().isoformat()
        rows.append(
            {
                "comparison_id": _comparison_id(source, str(row.get("video_id")), matched_date),
                "video_id": str(row.get("video_id", "")),
                "title": str(row.get("title", "")),
                "source": source,
                "published_date": pd.Timestamp(published).date().isoformat(),
                "matched_market_date": matched_date,
                "matched_source": str(matched.get("source", "")),
                "days_from_tradebot": int(
                    abs(
                        (
                            pd.Timestamp(published).normalize()
                            - pd.Timestamp(matched["market_date_dt"]).normalize()
                        ).days
                    )
                ),
                "macro_posture_score": macro_score,
                "macro_posture_label": str(row.get("macro_posture_label", "")),
                "classification_text_source": str(row.get("classification_text_source", "")),
                "classification_confidence": _optional_float(row.get("classification_confidence")),
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
                "trade_bot_base_defensive_weight": _optional_float(
                    matched.get("base_defensive_weight")
                ),
                "trade_bot_final_defensive_weight": _optional_float(
                    matched.get("final_defensive_weight")
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
        model.get("metadata", {}).get("lockupMetadataViewModel", {}).get("title", {}).get("content")
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


def _sort_manifest_frame(videos: pd.DataFrame) -> pd.DataFrame:
    if videos.empty:
        return videos
    output = videos.copy()
    if "published_date" in output:
        output["_published_sort"] = pd.to_datetime(output["published_date"], errors="coerce")
        output = output.sort_values(["_published_sort", "video_id"], ascending=[False, True])
        output = output.drop(columns=["_published_sort"])
    elif "video_id" in output:
        output = output.sort_values("video_id")
    expected = [
        "video_id",
        "source",
        "published_date",
        "title",
        "url",
        "transcript_path",
        "word_count",
        "fetched_at_utc",
        "status",
        "error",
    ]
    for column in expected:
        if column not in output:
            output[column] = ""
    return output[expected]


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
    return _parse_transcript_header_text(path.read_text(encoding="utf-8", errors="ignore"))


def _parse_transcript_header_text(text: str) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in text.splitlines()[:12]:
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


def _extract_video_id(text: str) -> str:
    for pattern in (
        r"(?:youtube\.com/watch\?v=|youtubetotranscript\.com/transcript\?v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"\b(?:video_id|video id)\s*:\s*([A-Za-z0-9_-]{11})\b",
        r"\b([A-Za-z0-9_-]{11})\b",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def _date_from_text_or_filename(text: str) -> str:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    return match.group(1) if match else ""


def _extract_youtube_transcript_page_title(text: str) -> str:
    plain = _html_to_plain_text(text)
    match = re.search(r"Transcript of\s+(.+?)(?:\n|Author\s*:)", plain, flags=re.S)
    if not match:
        return ""
    return _normalize_whitespace(match.group(1))


def _extract_manual_transcript_text(text: str) -> str:
    header = _parse_transcript_header_text(text)
    if header and "\n\n" in text:
        text = text.split("\n\n", 1)[1]
    plain = _html_to_plain_text(text)
    lines = [_normalize_whitespace(line) for line in plain.splitlines()]
    lines = [line for line in lines if line]
    start_index = 0
    for marker in ("Translate", "Timestamp OFF", "Copy"):
        marker_indexes = [index for index, line in enumerate(lines) if line == marker]
        if marker_indexes:
            start_index = max(start_index, marker_indexes[-1] + 1)
    cleaned: list[str] = []
    skip_prefixes = (
        "Transcript of ",
        "Author :",
        "Author:",
        "source:",
        "published_date:",
        "date:",
        "video_id:",
        "url:",
        "title:",
    )
    skip_exact = {
        "Transcript",
        "Copy",
        "Timestamp OFF",
        "Timestamp ON",
        "Translate",
        "AI Features",
        "Feedback",
        "Popular Features",
        "Output Language",
    }
    for line in lines[start_index:]:
        if line in skip_exact:
            continue
        if any(line.startswith(prefix) for prefix in skip_prefixes):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _html_to_plain_text(text: str) -> str:
    if "<" not in text or ">" not in text:
        return text
    without_scripts = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.I | re.S)
    with_breaks = re.sub(
        r"</?(p|div|br|h[1-6]|li|section|article)\b[^>]*>", "\n", without_scripts, flags=re.I
    )
    plain = re.sub(r"<[^>]+>", " ", with_breaks)
    plain = unescape(plain)
    return re.sub(r"\n{3,}", "\n\n", plain)


def _video_id_for_title(title: str, manifest_by_title: dict[str, dict[str, object]]) -> str:
    normalized = _normalize_title(title)
    if not normalized:
        return ""
    if normalized in manifest_by_title:
        return str(manifest_by_title[normalized].get("video_id", ""))
    for manifest_title, row in manifest_by_title.items():
        if normalized in manifest_title or manifest_title in normalized:
            return str(row.get("video_id", ""))
    return ""


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
        if (
            transcript_path is not None
            and transcript_path.exists()
            and not transcript_path.is_dir()
        ):
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
    # The final defensive allocation is the canonical expression of Trade Bot's
    # posture.  ``risk_budget_multiplier`` is only the scenario/risk-status
    # capacity gate and can remain near one even when the selected strategy is
    # substantially defensive.  Treating it as total exposure made a 64% BIL
    # allocation appear risk-on in the external comparison.
    final_defensive = _optional_float(row.get("final_defensive_weight"))
    if final_defensive is not None:
        return float(max(-1.0, min(1.0, 1.0 - (2.0 * final_defensive))))
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
    _write_alignment_diagnostic_outputs(output_dir, comparisons)


def _write_alignment_diagnostic_outputs(
    output_dir: Path,
    comparisons: pd.DataFrame,
) -> None:
    """Refresh the legacy drilldown files from the canonical comparison frame."""

    if comparisons.empty:
        return
    frame = comparisons.copy()
    transcript_backed = frame[
        frame["classification_text_source"].eq("transcript")
    ].copy()
    transcript_backed.to_csv(
        output_dir / "daily_transcript_backed_comparison.csv",
        index=False,
    )

    large_change = frame[frame["large_change_focus"].astype(bool)].copy()
    large_change = large_change.sort_values(
        ["abs_disagreement", "published_date"],
        ascending=[False, False],
    )
    large_change.to_csv(output_dir / "large_change_moments.csv", index=False)

    frame["year_month"] = pd.to_datetime(
        frame["published_date"], errors="coerce"
    ).dt.to_period("M").astype(str)
    monthly = (
        frame.groupby(["classification_text_source", "year_month"], dropna=False)
        .agg(
            rows=("video_id", "size"),
            macro_mean=("macro_posture_score", "mean"),
            trade_bot_mean=("trade_bot_posture_score", "mean"),
            mean_gap=("disagreement", "mean"),
            mean_abs_disagreement=("abs_disagreement", "mean"),
            major_mismatch_rate=(
                "disagreement_label",
                lambda values: float(pd.Series(values).eq("major_mismatch").mean()),
            ),
            large_change_rows=("large_change_focus", "sum"),
        )
        .reset_index()
    )
    monthly.to_csv(output_dir / "monthly_alignment.csv", index=False)

    aggregate_rows = []
    for scope, scoped in (
        ("all", frame),
        ("transcript_backed", transcript_backed),
        (
            "title_only",
            frame[frame["classification_text_source"].eq("title_only")],
        ),
    ):
        if scoped.empty:
            continue
        aggregate_rows.append(
            {
                "scope": scope,
                "rows": len(scoped),
                "date_min": str(scoped["published_date"].min()),
                "date_max": str(scoped["published_date"].max()),
                "macro_mean": float(scoped["macro_posture_score"].mean()),
                "trade_bot_mean": float(scoped["trade_bot_posture_score"].mean()),
                "macro_minus_trade_bot": float(scoped["disagreement"].mean()),
                "mean_abs_disagreement": float(scoped["abs_disagreement"].mean()),
                "major_mismatch_rate": float(
                    scoped["disagreement_label"].eq("major_mismatch").mean()
                ),
                "large_change_rows": int(scoped["large_change_focus"].sum()),
                "large_change_major_mismatch_rate": float(
                    scoped.loc[
                        scoped["large_change_focus"].astype(bool),
                        "disagreement_label",
                    ].eq("major_mismatch").mean()
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "aggregate_analysis.csv", index=False)
    (output_dir / "aggregate_analysis.md").write_text(
        _aggregate_alignment_markdown(aggregate),
        encoding="utf-8",
    )


def _aggregate_alignment_markdown(aggregate: pd.DataFrame) -> str:
    lines = [
        "# 42 Macro / Trade-Bot Aggregate Alignment Analysis",
        "",
        "This file is regenerated from the canonical comparison frame. The 42 Macro "
        "score is a lexical transcript proxy, not a reconstruction of its proprietary "
        "portfolio or horizon-specific model state.",
        "",
        "| scope | rows | dates | 42 mean | bot mean | mean abs gap | major mismatch |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row['scope']} | {int(row['rows'])} | {row['date_min']} to "
            f"{row['date_max']} | {row['macro_mean']:.2f} | "
            f"{row['trade_bot_mean']:.2f} | {row['mean_abs_disagreement']:.2f} | "
            f"{row['major_mismatch_rate']:.1%} |"
        )
    return "\n".join(lines) + "\n"


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
        "## Latest Transcript-Backed Read",
        "",
    ]
    latest = comparisons.sort_values(["published_date", "video_id"]).iloc[-1]
    final_defensive = _optional_float(latest.get("trade_bot_final_defensive_weight"))
    lines.extend(
        [
            f"- Video: {latest.get('published_date', '')} — {latest.get('title', '')}",
            f"- 42 Macro: {latest['macro_posture_label']} "
            f"({latest['macro_posture_score']:+.2f}).",
            f"- Trade Bot: {latest['trade_bot_posture_label']} "
            f"({latest['trade_bot_posture_score']:+.2f})"
            + (
                f", from a {final_defensive:.1%} final defensive allocation."
                if final_defensive is not None
                else "."
            ),
            f"- Comparison: {latest['disagreement_label']} "
            f"(absolute posture gap {latest['abs_disagreement']:.2f}).",
            "",
            "Trade Bot posture is derived from the final defensive allocation when that "
            "field is available. Risk-budget capacity alone is not total exposure.",
            "",
            "## Most Recent Videos",
            "",
            "These are mechanical lexical classifications. Mixed-horizon commentary "
            "must be read in context; the scalar is not a 42 Macro portfolio target.",
            "",
            "| date | video | 42 lexical proxy | Trade Bot | final defense | result |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    recent = comparisons.sort_values(["published_date", "video_id"]).tail(6)
    for _, row in recent.iterrows():
        row_defensive = _optional_float(row.get("trade_bot_final_defensive_weight"))
        defense_label = f"{row_defensive:.1%}" if row_defensive is not None else "n/a"
        title = str(row.get("title", "")).replace("|", "\\|")
        lines.append(
            f"| {row['published_date']} | [{title}]({_youtube_url(str(row['video_id']))}) | "
            f"{row['macro_posture_label']} ({row['macro_posture_score']:+.2f}) | "
            f"{row['trade_bot_posture_label']} ({row['trade_bot_posture_score']:+.2f}) | "
            f"{defense_label} | {row['disagreement_label']} |"
        )
    lines.extend(
        [
            "",
        "## Largest Mismatches",
        "",
        ]
    )
    if comparisons.empty:
        lines.append("No comparisons were produced.")
        return "\n".join(lines) + "\n"
    top = comparisons.sort_values("abs_disagreement", ascending=False).head(10)
    lines.append("| date | video_id | 42 Macro | trade-bot | gap | large-change | note |")
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
