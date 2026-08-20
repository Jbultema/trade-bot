#!/usr/bin/env python3
"""Build the released AI-financial-system database from its own release tree."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def insert_csv(
    connection: sqlite3.Connection,
    table: str,
    path: Path,
) -> None:
    fields, rows = read_csv(path)
    table_info = {
        item[1]: {
            "notnull": bool(item[3]),
            "primary_key": bool(item[5]),
        }
        for item in connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    }
    if set(fields) != set(table_info):
        raise RuntimeError(
            f"{table}: CSV/schema mismatch {set(fields) ^ set(table_info)}"
        )
    columns = ", ".join(f'"{field}"' for field in fields)
    placeholders = ", ".join("?" for _ in fields)
    connection.executemany(
        f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders})',
        (
            [
                (
                    row[field]
                    if row[field]
                    or table_info[field]["notnull"]
                    or table_info[field]["primary_key"]
                    else None
                )
                for field in fields
            ]
            for row in rows
        ),
    )


def load_plan(root: Path) -> dict[str, Any]:
    plan = json.loads(
        (root / "data/convergence/database_load_plan.json").read_text(
            encoding="utf-8"
        )
    )
    if plan.get("version") != 1:
        raise RuntimeError("unsupported database load-plan version")
    return plan


def build(root: Path, output: Path) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"release root missing: {root}")
    plan = load_plan(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.db")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.executescript(
            (root / "schema/canonical_core.sql").read_text(
                encoding="utf-8"
            )
        )
        for item in plan["core_tables"]:
            insert_csv(
                connection,
                item["table"],
                root / item["relative_path"],
            )
        connection.executescript(
            (root / "schema/convergence_extensions.sql").read_text(
                encoding="utf-8"
            )
        )
        for item in plan["extension_tables"]:
            insert_csv(
                connection,
                item["table"],
                root / item["relative_path"],
            )
        connection.executescript(
            """
            CREATE VIEW claim_evidence AS
            SELECT
                c.claim_id,
                c.chapter_id,
                c.claim_text,
                c.evidence_grade AS claim_grade,
                c.confidence,
                s.source_id,
                s.evidence_grade AS source_grade,
                s.title,
                s.publisher,
                s.url,
                e.locator,
                e.support_type
            FROM claims c
            LEFT JOIN evidence_links e ON e.claim_id = c.claim_id
            LEFT JOIN sources s ON s.source_id = e.source_id;

            CREATE VIEW unresolved_high_materiality AS
            SELECT
                unknown_id,
                chapter_id,
                question,
                materiality,
                unknown_class,
                downstream_blocking_chapters,
                resolution_trigger,
                suspected_sources,
                resolution_plan,
                status,
                last_reviewed
            FROM unknowns
            WHERE status <> 'resolved' AND materiality = 'high';
            """
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        select_star_views = connection.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'view' AND lower(sql) LIKE '%select *%'
            """
        ).fetchone()[0]
        if integrity != "ok" or foreign_keys or select_star_views:
            raise RuntimeError(
                {
                    "integrity": integrity,
                    "foreign_keys": foreign_keys,
                    "select_star_views": select_star_views,
                }
            )
    finally:
        connection.close()
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.release_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
