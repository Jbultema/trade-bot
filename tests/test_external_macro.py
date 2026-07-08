from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.external_macro import (
    build_macro_tradebot_comparisons,
    classify_42macro_transcript,
    summarize_macro_alignment,
)
from trade_bot.storage.warehouse import TradingWarehouse


def test_42macro_classifier_distinguishes_constructive_and_defensive_reads() -> None:
    constructive = classify_42macro_transcript(
        "The macro weather model has a bullish outlook and a risk-on market regime. "
        "Liquidity tailwinds support the rally.",
        title="How much runway does this bull market have left?",
        published_date="2026-07-06",
        video_id="abc",
    )
    defensive = classify_42macro_transcript(
        "The rising probability of a risk-off market regime should cause investors "
        "to reduce gross exposure, book gains, and use winners as a source of funds.",
        title="Should investors stop using the Mag-7 as a source of funds?",
        published_date="2026-07-08",
        video_id="def",
    )

    assert constructive["macro_posture_score"] > 0.4
    assert constructive["macro_posture_label"] == "risk_on"
    assert defensive["macro_posture_score"] < -0.4
    assert defensive["macro_posture_label"] == "risk_reduction"
    assert defensive["large_change_flag"] is True


def test_macro_tradebot_comparison_matches_nearest_operating_date() -> None:
    classifications = pd.DataFrame(
        [
            {
                "classification_id": "42:abc:2026-07-07",
                "video_id": "abc",
                "source": "42macro_youtube",
                "published_date": "2026-07-07",
                "title": "Test",
                "macro_posture_score": 0.80,
                "macro_posture_label": "risk_on",
                "near_term_risk_score": 0.1,
                "medium_term_bullish_score": 0.9,
                "large_change_flag": True,
                "key_themes": "ai",
                "classified_at_utc": "2026-07-08T00:00:00+00:00",
            }
        ]
    )
    operating_metrics = pd.DataFrame(
        [
            {
                "history_id": "reconstructed:2026-07-08",
                "history_time": "2026-07-08",
                "snapshot_time": "2026-07-08",
                "market_date": "2026-07-08",
                "run_id": "reconstructed:2026-07-08",
                "source": "reconstructed_price_fast_point_in_time",
                "reconstruction_note": "test",
                "risk_score": 0.70,
                "one_month_risk_off_probability": 0.45,
                "risk_budget_multiplier": 0.25,
                "portfolio_risk_multiplier": 0.25,
            }
        ]
    )

    comparisons = build_macro_tradebot_comparisons(
        classifications,
        operating_metrics,
        max_match_days=3,
    )
    summary = summarize_macro_alignment(classifications, comparisons)

    assert len(comparisons) == 1
    assert comparisons.iloc[0]["matched_market_date"] == "2026-07-08"
    assert comparisons.iloc[0]["trade_bot_posture_label"] == "defensive"
    assert comparisons.iloc[0]["disagreement_label"] == "major_mismatch"
    assert bool(comparisons.iloc[0]["large_change_focus"])
    assert summary["major_mismatches"] == 1


def test_warehouse_persists_external_macro_alignment(tmp_path) -> None:
    warehouse = TradingWarehouse(tmp_path / "trade_bot.duckdb")
    videos = pd.DataFrame(
        [
            {
                "video_id": "abc",
                "source": "42macro_youtube",
                "published_date": "2026-07-07",
                "title": "Test",
                "url": "https://www.youtube.com/watch?v=abc",
                "transcript_path": "2026-07-07_abc_test.txt",
                "word_count": 100,
                "fetched_at_utc": "2026-07-08T00:00:00+00:00",
                "status": "fetched",
                "error": "",
            }
        ]
    )
    classifications = pd.DataFrame(
        [
            {
                "classification_id": "42macro_youtube:2026-07-07:abc",
                "video_id": "abc",
                "source": "42macro_youtube",
                "published_date": "2026-07-07",
                "title": "Test",
                "macro_posture_score": 0.7,
                "macro_posture_label": "risk_on",
                "near_term_risk_score": 0.1,
                "medium_term_bullish_score": 0.9,
                "large_change_flag": True,
                "bullish_term_score": 3.0,
                "defensive_term_score": 0.2,
                "key_themes": "ai",
                "classified_at_utc": "2026-07-08T00:00:00+00:00",
            }
        ]
    )
    comparisons = pd.DataFrame(
        [
            {
                "comparison_id": "42macro_youtube:abc:2026-07-08",
                "video_id": "abc",
                "source": "42macro_youtube",
                "published_date": "2026-07-07",
                "matched_market_date": "2026-07-08",
                "matched_source": "reconstructed_price_fast_point_in_time",
                "days_from_tradebot": 1,
                "macro_posture_score": 0.7,
                "macro_posture_label": "risk_on",
                "trade_bot_posture_score": -0.4,
                "trade_bot_posture_label": "defensive",
                "disagreement": 1.1,
                "abs_disagreement": 1.1,
                "disagreement_label": "major_mismatch",
                "large_change_focus": True,
                "trade_bot_risk_score": 0.7,
                "trade_bot_risk_budget_multiplier": 0.25,
                "trade_bot_risk_off_probability": 0.45,
                "trade_bot_portfolio_risk_multiplier": 0.25,
                "notes": "test",
                "compared_at_utc": "2026-07-08T00:00:00+00:00",
            }
        ]
    )

    counts = warehouse.save_external_macro_alignment(
        videos=videos,
        classifications=classifications,
        comparisons=comparisons,
    )

    assert counts["external_macro_videos"] == 1
    assert warehouse.read_table("external_macro_videos").iloc[0]["video_id"] == "abc"
    assert (
        warehouse.read_table("external_macro_classifications").iloc[0][
            "macro_posture_label"
        ]
        == "risk_on"
    )
    assert warehouse.read_table("external_macro_tradebot_comparisons").iloc[0][
        "disagreement"
    ] == pytest.approx(1.1)
