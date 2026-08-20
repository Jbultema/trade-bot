from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DATABASE = REPO_ROOT / "reports/ai_financial_system/research.db"
OUTPUT = REPO_ROOT / "reports/ai_financial_system/chapters/chapter_01/data"
CHAPTER_HTML = PROJECT_ROOT / "chapters/chapter_01/chapter.html"


WRITING_GATE_AUDIT = [
    {
        "gate_id": "WG-0101",
        "gate": "Evidence grades and confidence are defined separately",
        "status": "pass",
        "evidence": "Chapter 1 defines Grades A through D and five confidence states.",
    },
    {
        "gate_id": "WG-0102",
        "gate": "Transaction status and amount semantics are explicit",
        "status": "pass",
        "evidence": "Status ladder and stock flow ceiling floor contingent and physical quantity definitions are visible.",
    },
    {
        "gate_id": "WG-0103",
        "gate": "Accounting observations and model estimates remain separate",
        "status": "pass",
        "evidence": "Model estimates retain assumptions and do not inherit primary-source evidence grade.",
    },
    {
        "gate_id": "WG-0104",
        "gate": "Return and first-loss measures are defined",
        "status": "pass",
        "evidence": "Project IRR accounting ROIC incremental ROIC economic return and first-loss threshold are distinct.",
    },
    {
        "gate_id": "WG-0105",
        "gate": "Utilization and demand states are defined",
        "status": "pass",
        "evidence": "Installed-through-productive ladder and financial physical and economic demand are explicit.",
    },
    {
        "gate_id": "WG-0106",
        "gate": "Exposure identity and nonadditive component rules are defined",
        "status": "pass",
        "evidence": "Stable exposure state vector and downstream component joins prevent cross-chapter double counting.",
    },
    {
        "gate_id": "WG-0107",
        "gate": "Falsification missing evidence confidence evolution and importance are visible",
        "status": "pass",
        "evidence": "All four publication-control sections are present and backed by structured rows.",
    },
    {
        "gate_id": "WG-0108",
        "gate": "Chapter handoffs and blocked inputs are governed",
        "status": "pass",
        "evidence": "Schema-v2 handoff register preserves ready bounded blocked and not-started states.",
    },
]


def query(connection: sqlite3.Connection, sql: str) -> list[dict[str, object]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(sql).fetchall()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty artifact: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    connection = sqlite3.connect(DATABASE)
    claims = query(
        connection,
        """
        SELECT c.claim_id,c.claim_text,c.evidence_grade,c.confidence,c.claim_type,
               c.accounting_status,c.economic_status,c.quantification_status,
               c.effective_date,GROUP_CONCAT(e.source_id,';') AS source_ids
        FROM claims c
        LEFT JOIN evidence_links e ON e.claim_id=c.claim_id
        WHERE c.chapter_id='CH01'
        GROUP BY c.claim_id
        ORDER BY c.claim_id
        """,
    )
    unknowns = query(
        connection,
        "SELECT * FROM unknowns WHERE chapter_id='CH01' ORDER BY unknown_id",
    )
    confidence = query(
        connection,
        """
        SELECT h.* FROM claim_confidence_history h
        JOIN claims c ON c.claim_id=h.claim_id
        WHERE c.chapter_id='CH01'
        ORDER BY h.history_id
        """,
    )
    importance = query(
        connection,
        """
        SELECT i.* FROM claim_economic_importance i
        JOIN claims c ON c.claim_id=i.claim_id
        WHERE c.chapter_id='CH01'
        ORDER BY i.importance_id
        """,
    )
    source_mismatches = query(
        connection,
        """
        SELECT c.claim_id
        FROM claims c
        LEFT JOIN evidence_links e ON e.claim_id=c.claim_id
        LEFT JOIN sources s ON s.source_id=e.source_id
        WHERE c.chapter_id='CH01' AND c.evidence_grade='A'
        GROUP BY c.claim_id
        HAVING SUM(CASE WHEN s.evidence_grade='A' THEN 1 ELSE 0 END)=0
        """,
    )
    handoff_statuses = query(
        connection,
        """
        SELECT readiness_status,COUNT(*) AS handoff_count
        FROM chapter_handoffs
        GROUP BY readiness_status
        ORDER BY readiness_status
        """,
    )
    foreign_keys = query(connection, "PRAGMA foreign_key_check")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()

    expected = {
        "claims": 8,
        "unknowns": 4,
        "confidence_history": 8,
        "economic_importance": 8,
        "writing_gate": 8,
        "chapter_handoffs": 17,
    }
    actual = {
        "claims": len(claims),
        "unknowns": len(unknowns),
        "confidence_history": len(confidence),
        "economic_importance": len(importance),
        "writing_gate": len(WRITING_GATE_AUDIT),
        "chapter_handoffs": sum(int(row["handoff_count"]) for row in handoff_statuses),
    }
    if actual != expected:
        raise ValueError(f"Unexpected Chapter 1 publication counts: {actual}")
    if source_mismatches:
        raise ValueError(f"Grade-A claims without Grade-A sources: {source_mismatches}")
    if foreign_keys or integrity != "ok":
        raise ValueError(f"Database validation failed: integrity={integrity}; fk={foreign_keys}")
    if any(row["prior_evidence_grade"] or row["prior_confidence"] for row in confidence):
        raise ValueError("Chapter 1 baseline must not invent prior confidence history")
    if any(row["status"] != "pass" for row in WRITING_GATE_AUDIT):
        raise ValueError("Chapter 1 writing gate has not passed")

    chapter_text = " ".join(CHAPTER_HTML.read_text(encoding="utf-8").split())
    required_phrases = [
        "Evidence that would overturn",
        "Evidence still missing",
        "Confidence evolution",
        "Economic importance",
        "Project IRR",
        "Accounting ROIC",
        "Incremental ROIC",
        "first-loss threshold",
        "financial demand",
        "physical demand",
        "economic demand",
        "common exposure state vector",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in chapter_text]
    if missing:
        raise ValueError(f"Chapter 1 required methodology text missing: {missing}")
    database_hash = sha256(DATABASE)
    if database_hash not in chapter_text:
        raise ValueError("Chapter 1 does not embed the current database hash")

    outputs = {
        "chapter_01_claims.csv": claims,
        "chapter_01_unknowns.csv": unknowns,
        "claim_confidence_history.csv": confidence,
        "claim_economic_importance.csv": importance,
        "chapter_handoff_statuses.csv": handoff_statuses,
        "writing_gate_audit.csv": WRITING_GATE_AUDIT,
    }
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    for filename, rows in outputs.items():
        write_csv(OUTPUT / filename, rows)

    manifest = {
        "chapter": 1,
        "schema_version": 2,
        "database": str(DATABASE.relative_to(REPO_ROOT)),
        "database_sha256": database_hash,
        "counts": actual,
        "artifacts": {
            filename: {"sha256": sha256(OUTPUT / filename), "rows": len(rows)}
            for filename, rows in outputs.items()
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
