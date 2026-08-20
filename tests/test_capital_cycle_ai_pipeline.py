from __future__ import annotations

import json

import pandas as pd

from trade_bot.research.capital_cycle_ai_pipeline import (
    audit_ai_family_coverage,
    fetch_sec_companyfacts_archive,
)


class _Response:
    status_code = 200
    content = b'{"facts": {"us-gaap": {}}}'

    def raise_for_status(self) -> None:
        return None


def test_sec_fetcher_archives_official_payload_and_hashes_it(tmp_path, monkeypatch) -> None:
    seen = []

    def fake_get(self, url, timeout):  # noqa: ANN001
        seen.append((url, timeout, self.headers["User-Agent"]))
        return _Response()

    monkeypatch.setattr("requests.Session.get", fake_get)
    entities = pd.DataFrame(
        [{"episode": "ai_buildout_2022_present", "entity": "Example", "cik": 123}]
    )

    inventory = fetch_sec_companyfacts_archive(
        entity_manifest=entities,
        research_dir=tmp_path,
        user_agent="Example Research example@example.com",
    )

    assert inventory.iloc[0]["status"] == "downloaded"
    assert inventory.iloc[0]["sha256"]
    raw_path = tmp_path / "raw" / "sec_companyfacts" / "CIK0000000123.json"
    assert json.loads(raw_path.read_text()) == {"facts": {"us-gaap": {}}}
    assert seen[0][0].endswith("CIK0000000123.json")
    assert seen[0][2] == "Example Research example@example.com"


def test_ai_coverage_requires_freshness_rows_and_entity_breadth() -> None:
    observations = pd.DataFrame(
        [
            {
                "family": "demand",
                "entity": entity,
                "source_id": f"source_{entity}",
                "available_date": "2026-07-01",
                "eligible_for_model": True,
            }
            for entity in ("A", "B")
        ]
        + [
            {
                "family": "capacity",
                "entity": "A",
                "source_id": "old",
                "available_date": "2024-01-01",
                "eligible_for_model": True,
            }
        ]
    )

    audit = audit_ai_family_coverage(observations, as_of="2026-08-20").set_index("family")

    assert bool(audit.loc["demand", "available"])
    assert bool(audit.loc["capacity", "observed"])
    assert not bool(audit.loc["capacity", "available"])
    assert "stale_or_undated" in audit.loc["capacity", "missing_reason"]
