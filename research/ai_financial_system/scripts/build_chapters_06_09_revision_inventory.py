from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DATABASE = REPO_ROOT / "reports/ai_financial_system/research.db"
SOURCE_ROOT = PROJECT_ROOT / "sources"
OUTPUT = PROJECT_ROOT / "staging/chapters_06_09_source_preservation_inventory.csv"
VALIDATION = (
    PROJECT_ROOT / "staging/chapters_06_09_source_preservation_inventory_validation.json"
)
CHAPTERS = ("CH06", "CH07", "CH08", "CH09")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preservation_class(source_type: str, primary_document: int) -> str:
    source_type_lower = source_type.casefold()
    if primary_document and (
        source_type_lower.startswith("sec")
        or "official" in source_type_lower
        or "regulatory" in source_type_lower
    ):
        return "P1"
    if primary_document:
        return "P2"
    return "P3"


def local_originals(source_id: str) -> list[Path]:
    matches = []
    for path in SOURCE_ROOT.rglob(f"{source_id}_*"):
        if not path.is_file():
            continue
        name = path.name.casefold()
        if (
            "visible_text" in name
            or "derived" in name
            or "validation" in name
            or name.endswith(".response.json")
            or name == "source_archive_manifest.json"
        ):
            continue
        matches.append(path)
    return sorted(matches)


def main() -> None:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in CHAPTERS)
    rows = connection.execute(
        f"""
        SELECT
            s.source_id,
            s.evidence_grade AS source_grade,
            s.title,
            s.publisher,
            s.source_type,
            s.url,
            s.publication_date,
            s.retrieved_at,
            s.primary_document,
            s.archived_path,
            s.content_sha256,
            s.evidentiary_scope,
            s.limitations,
            c.chapter_id,
            c.claim_id,
            c.evidence_grade AS claim_grade,
            c.confidence,
            el.locator,
            el.support_type
        FROM claims c
        JOIN evidence_links el ON el.claim_id = c.claim_id
        JOIN sources s ON s.source_id = el.source_id
        WHERE c.chapter_id IN ({placeholders})
        ORDER BY s.source_id, c.chapter_id, c.claim_id, el.locator
        """,
        CHAPTERS,
    ).fetchall()
    connection.close()

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)

    output_rows = []
    for source_id, source_rows in sorted(grouped.items()):
        first = source_rows[0]
        originals = local_originals(source_id)
        archive_paths = [
            str(path.relative_to(REPO_ROOT))
            for path in originals
        ]
        archive_hashes = [sha256(path) for path in originals]
        archive_sizes = [str(path.stat().st_size) for path in originals]
        pclass = preservation_class(
            first["source_type"], int(first["primary_document"])
        )
        if originals:
            required_action = "verify_existing_archive_and_build_atomic_rebase"
        elif pclass in {"P1", "P2"}:
            required_action = "preserve_source_or_retain_claim_level_hold"
        else:
            required_action = "record_rights_boundary_and_permitted_locator"
        output_rows.append(
            {
                "source_id": source_id,
                "chapters": "|".join(
                    sorted({row["chapter_id"] for row in source_rows})
                ),
                "claim_ids": "|".join(
                    sorted({row["claim_id"] for row in source_rows})
                ),
                "claim_count": len({row["claim_id"] for row in source_rows}),
                "evidence_link_count": len(source_rows),
                "claim_grades": "|".join(
                    sorted({row["claim_grade"] for row in source_rows})
                ),
                "confidence_states": "|".join(
                    sorted({row["confidence"] for row in source_rows})
                ),
                "support_types": "|".join(
                    sorted({row["support_type"] for row in source_rows})
                ),
                "locators": " || ".join(
                    sorted({row["locator"] for row in source_rows})
                ),
                "source_grade": first["source_grade"],
                "title": first["title"],
                "publisher": first["publisher"],
                "source_type": first["source_type"],
                "url": first["url"] or "",
                "publication_date": first["publication_date"] or "",
                "retrieved_at": first["retrieved_at"],
                "primary_document": int(first["primary_document"]),
                "preservation_class": pclass,
                "canonical_archived_path": first["archived_path"] or "",
                "canonical_content_sha256": first["content_sha256"] or "",
                "local_original_paths": "|".join(archive_paths),
                "local_original_sha256": "|".join(archive_hashes),
                "local_original_sizes": "|".join(archive_sizes),
                "local_original_count": len(originals),
                "evidentiary_scope": first["evidentiary_scope"],
                "limitations": first["limitations"],
                "required_action": required_action,
                "semantic_locator_review_status": "not_reperformed_by_inventory",
                "publication_status": "held_pending_preservation_and_locator_review",
            }
        )

    fieldnames = list(output_rows[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    primary_count = sum(row["primary_document"] for row in output_rows)
    existing_archive_count = sum(bool(row["local_original_count"]) for row in output_rows)
    validation = {
        "status": "PASS",
        "database": str(DATABASE.relative_to(REPO_ROOT)),
        "database_sha256": sha256(DATABASE),
        "chapters": list(CHAPTERS),
        "source_count": len(output_rows),
        "primary_source_count": primary_count,
        "canonical_archived_source_count": sum(
            bool(row["canonical_archived_path"]) for row in output_rows
        ),
        "local_original_source_count": existing_archive_count,
        "unpreserved_source_count": len(output_rows) - existing_archive_count,
        "claim_count": len({row["claim_id"] for row in rows}),
        "evidence_link_count": len(rows),
        "output": str(OUTPUT.relative_to(REPO_ROOT)),
        "output_sha256": sha256(OUTPUT),
        "boundary": (
            "Inventory and byte-presence check only. A local file or hash does not "
            "establish semantic locator support or authorize publication."
        ),
    }
    VALIDATION.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
