from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "research/ai_financial_system/scripts/preserve_public_source_set.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("preserve_public_source_set", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_evidence_grades_map_to_preservation_classes(tmp_path: Path) -> None:
    inventory = tmp_path / "sources.csv"
    inventory.write_text(
        "source_id,evidence_grade,title,publisher,url,publication_date\n"
        "SRC-0001,A,Primary,Issuer,https://example.com/a,2026-01-01\n"
        "SRC-0002,B,Reporting,Publisher,https://example.com/b,2026-01-02\n"
        "SRC-0003,C,Inference,Analyst,https://example.com/c,2026-01-03\n",
        encoding="utf-8",
    )

    rows = _module().load_rows(inventory, set())

    assert [row["preservation_class"] for row in rows] == ["P1", "P2", "P4"]


def test_resume_record_matches_source_id_and_exact_url(tmp_path: Path) -> None:
    source_id = "SRC-0110"
    first_url = "https://example.com/financial-statements.pdf"
    second_url = "https://example.com/management-report.pdf"
    archived = tmp_path / f"{source_id}_abc.pdf"
    archived.write_bytes(b"first")
    metadata = tmp_path / f"{source_id}_abc.response.json"
    metadata.write_text(
        json.dumps(
            {
                "source_id": source_id,
                "original_url": first_url,
                "status": "preserved_complete",
                "content_sha256": "abc",
                "local_archive_path": str(archived),
            }
        ),
        encoding="utf-8",
    )

    module = _module()

    assert module.latest_existing_record(tmp_path, source_id, first_url) is not None
    assert module.latest_existing_record(tmp_path, source_id, second_url) is None
