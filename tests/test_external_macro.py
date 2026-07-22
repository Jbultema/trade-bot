from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research import external_macro
from trade_bot.research.external_macro import (
    build_forward_outcome_scores,
    build_macro_tradebot_comparisons,
    classify_42macro_transcript,
    import_42macro_transcript_files,
    summarize_forward_outcome_scores,
    summarize_macro_alignment,
    write_missing_42macro_transcript_priority,
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


def test_macro_tradebot_comparison_uses_final_allocation_as_posture() -> None:
    classifications = pd.DataFrame(
        [
            {
                "video_id": "current",
                "published_date": "2026-07-21",
                "macro_posture_score": -0.15,
                "macro_posture_label": "constructive_but_fragile",
                "classification_text_source": "transcript",
                "classification_confidence": 1.0,
                "large_change_flag": True,
            }
        ]
    )
    operating_metrics = pd.DataFrame(
        [
            {
                "market_date": "2026-07-21",
                "source": "point_in_time",
                "risk_score": 0.433333,
                "risk_budget_multiplier": 0.9,
                "one_month_risk_off_probability": 0.249,
                "portfolio_risk_multiplier": 1.0,
                "base_defensive_weight": 0.597,
                "final_defensive_weight": 0.63727,
            }
        ]
    )

    comparisons = build_macro_tradebot_comparisons(classifications, operating_metrics)

    row = comparisons.iloc[0]
    assert row["trade_bot_posture_score"] == pytest.approx(-0.27454)
    assert row["trade_bot_posture_label"] == "cautious"
    assert row["trade_bot_final_defensive_weight"] == pytest.approx(0.63727)


def test_alignment_diagnostics_refresh_legacy_drilldowns(tmp_path) -> None:
    comparisons = pd.DataFrame(
        [
            {
                "video_id": "recent",
                "published_date": "2026-07-21",
                "classification_text_source": "transcript",
                "macro_posture_score": -0.15,
                "trade_bot_posture_score": -0.27,
                "disagreement": 0.12,
                "abs_disagreement": 0.12,
                "disagreement_label": "aligned",
                "large_change_focus": True,
            }
        ]
    )

    external_macro._write_alignment_diagnostic_outputs(tmp_path, comparisons)

    daily = pd.read_csv(tmp_path / "daily_transcript_backed_comparison.csv")
    aggregate = pd.read_csv(tmp_path / "aggregate_analysis.csv")
    monthly = pd.read_csv(tmp_path / "monthly_alignment.csv")
    assert daily.iloc[0]["published_date"] == "2026-07-21"
    assert aggregate.iloc[0]["date_max"] == "2026-07-21"
    assert monthly.iloc[0]["year_month"] == "2026-07"


def test_manual_transcript_import_accepts_youtubetotranscript_copy(tmp_path) -> None:
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    (transcript_dir / "manifest.json").write_text(
        """
[
  {
    "video_id": "lxZyYBLbKnw",
    "source": "42macro_youtube",
    "published_date": "2026-07-08",
    "title": "Should investors stop using the Mag-7 as a Source of Funds?",
    "url": "https://www.youtube.com/watch?v=lxZyYBLbKnw",
    "transcript_path": "",
    "word_count": 0,
    "fetched_at_utc": "",
    "status": "catalog_only",
    "error": ""
  }
]
""",
        encoding="utf-8",
    )
    input_dir = tmp_path / "manual"
    input_dir.mkdir()
    (input_dir / "lxZyYBLbKnw.txt").write_text(
        """
Transcript of Should investors stop using the Mag-7 as a Source of Funds?
Author : 42 Macro
Transcript
Copy
Timestamp OFF
Translate

Happy Wednesday out there team42. The rising probability of a risk-off market
regime should cause investors to reduce gross exposure and book gains. Winners
may become a source of funds if volatility rises. This is tactical risk
reduction, not a long-term bearish view.
""",
        encoding="utf-8",
    )

    result = import_42macro_transcript_files(
        input_dir=input_dir,
        transcript_dir=transcript_dir,
    )

    assert result.imported == 1
    row = result.videos.iloc[0]
    assert row["video_id"] == "lxZyYBLbKnw"
    assert row["status"] == "imported_manual"
    transcript_path = transcript_dir / str(row["transcript_path"])
    assert transcript_path.exists()
    assert "Timestamp OFF" not in transcript_path.read_text(encoding="utf-8")
    assert "risk-off market" in transcript_path.read_text(encoding="utf-8")


def test_missing_transcript_priority_prefers_large_change_rows(tmp_path) -> None:
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    (transcript_dir / "manifest.json").write_text(
        """
[
  {
    "video_id": "missing1234",
    "source": "42macro_youtube",
    "published_date": "2026-07-08",
    "title": "Risk-off warning",
    "url": "https://www.youtube.com/watch?v=missing1234",
    "transcript_path": "",
    "word_count": 0,
    "fetched_at_utc": "",
    "status": "catalog_only",
    "error": ""
  }
]
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "video_id": "missing1234",
                "published_date": "2026-07-08",
                "large_change_focus": True,
                "abs_disagreement": 0.8,
            }
        ]
    ).to_csv(output_dir / "daily_comparison.csv", index=False)

    priority = write_missing_42macro_transcript_priority(
        transcript_dir=transcript_dir,
        output_dir=output_dir,
    )

    assert len(priority) == 1
    assert priority.iloc[0]["priority_score"] >= 100
    assert priority.iloc[0]["transcript_url"].endswith("v=missing1234")
    assert "large-change" in priority.iloc[0]["priority_reason"]


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
        warehouse.read_table("external_macro_classifications").iloc[0]["macro_posture_label"]
        == "risk_on"
    )
    assert warehouse.read_table("external_macro_tradebot_comparisons").iloc[0][
        "disagreement"
    ] == pytest.approx(1.1)


def test_forward_outcome_scores_reward_defense_before_left_tail() -> None:
    dates = pd.bdate_range("2026-01-01", periods=30)
    prices = pd.DataFrame(
        {
            "SPY": [
                100,
                99,
                96,
                94,
                93,
                92,
                91,
                91,
                92,
                93,
                *([93] * 20),
            ],
            "BIL": [100 + index * 0.01 for index in range(30)],
        },
        index=dates,
    )
    comparisons = pd.DataFrame(
        [
            {
                "video_id": "abc",
                "published_date": "2026-01-01",
                "matched_market_date": "2026-01-01",
                "macro_posture_score": -0.8,
                "trade_bot_posture_score": 0.8,
                "classification_text_source": "transcript",
                "classification_confidence": 1.0,
            }
        ]
    )

    outcomes = build_forward_outcome_scores(
        comparisons,
        prices,
        horizons={"1w": 5},
    )
    summary = summarize_forward_outcome_scores(outcomes)

    assert outcomes.iloc[0]["realized_environment"] == "left_tail"
    assert outcomes.iloc[0]["macro_action_score"] > outcomes.iloc[0]["trade_bot_action_score"]
    assert bool(outcomes.iloc[0]["trade_bot_overrisk"])
    assert not bool(outcomes.iloc[0]["macro_overrisk"])
    assert (
        summary[summary["scope"].eq("transcript")].iloc[0]["macro_mean_action_score"]
        > summary[summary["scope"].eq("transcript")].iloc[0]["trade_bot_mean_action_score"]
    )
