"""Current-vintage public macro context, deliberately excluded from historical signals."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trade_bot.data.fred_data import load_fred_catalog


def macro_context():
    path = Path("data/cache/fred_series.parquet")
    captured = datetime.now(UTC)
    result = {"captured_at_utc": captured.isoformat(), "status": "unavailable",
              "vintage": "current_revised_not_point_in_time", "historical_replay_allowed": False,
              "publication_times_verified": False, "series": [],
              "limitations": ["Observation dates are not release dates; publication timestamps are not verified.",
                              "Current revised values cannot establish what was known at a past close.",
                              "Changes compare distinct observations, not exchange sessions or surprises versus consensus."]}
    if not path.exists():
        return result
    import hashlib

    result["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_parquet(path).loc[:pd.Timestamp(captured.date())]
    for entry in load_fred_catalog("configs/macro_fred.yaml"):
        values = frame[entry.series_id].dropna() if entry.series_id in frame else pd.Series(dtype=float)
        records = [{"observation_date": str(d.date()), "value": float(v)} for d, v in values.tail(30).items() if math.isfinite(v)]
        result["series"].append({"id": entry.series_id, "name": entry.name, "category": entry.category,
                                 "source_url": f"https://fred.stlouisfed.org/series/{entry.series_id}",
                                 "observations": records})
    result["status"] = "captured"
    return result


def external_research_context():
    path = Path("reports/42macro_alignment/daily_comparison.csv")
    context = {"status": "unavailable", "captured_at_utc": datetime.now(UTC).isoformat(), "rows": [],
               "limitations": ["Mechanical language classifications are not the author's actual portfolio or a human interpretation.",
                               "Date-level retrospective matching does not establish intraday information availability.",
                               "The legacy September current_human_read document has not been refreshed and is not current."]}
    if not path.exists():
        return context
    import hashlib

    frame = pd.read_csv(path).sort_values(["published_date", "video_id"]).tail(5)
    columns = [c for c in ("published_date", "video_id", "title", "matched_market_date", "matched_source",
                          "days_from_tradebot", "classification_text_source", "macro_posture_label",
                          "trade_bot_posture_label", "trade_bot_final_defensive_weight") if c in frame]
    context |= {"status": "captured", "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": json.loads(frame[columns].to_json(orient="records"))}
    return context
