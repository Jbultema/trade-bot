from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_CANDIDATE = PROJECT_ROOT / "integration/private_investment_candidate"
DEFAULT_RESULTS = (
    PROJECT_ROOT / "integration/private_investment_candidate_validation.json"
)
BUILDER = PROJECT_ROOT / "scripts/build_private_investment_candidate.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing header: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"CSV width mismatch: {path}")
    return rows


def index(
    rows: list[dict[str, str]], key: str, label: str
) -> dict[str, dict[str, str]]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key} in {label}")
    if "" in result:
        raise ValueError(f"blank {key} in {label}")
    return result


def number(value: str) -> float | None:
    return None if value == "" else float(value)


def require_close(actual: float, expected: float, label: str, tolerance: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label}: {actual} != {expected}")


def validate_manifest(candidate: Path) -> dict[str, Any]:
    manifest_path = candidate / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "BOUNDED_INTEGRATION_CANDIDATE_NOT_CANONICAL":
        raise ValueError("candidate status is not fail-closed")
    if (
        manifest["canonical_authorization"]
        != "AUTHORIZED_FOR_PRIMARY_AGENT_BOUNDED_REBASE_ONLY_WITH_RETAINED_HOLDS"
    ):
        raise ValueError("candidate lacks bounded acceptance authorization")
    required_holds = {
        "all eight primary-document snapshots and content hashes",
        "governed top-20 population reconstruction",
        "S&P 500 aggregate reproduction",
        "unnamed investee allocation at Alphabet and NVIDIA and residual Amazon attribution",
        "exact Microsoft component and tax allocation",
        "prospective Meta/Tesla and realization monitoring",
    }
    if set(manifest["retained_holds"]) != required_holds:
        raise ValueError("retained holds changed")
    for file_name, declared in manifest["files"].items():
        path = candidate / file_name
        if not path.is_file():
            raise ValueError(f"missing manifested file: {file_name}")
        if sha256(path) != declared["sha256"]:
            raise ValueError(f"hash mismatch: {file_name}")
        if path.suffix == ".csv" and len(read_csv(path)) != declared["rows"]:
            raise ValueError(f"row-count mismatch: {file_name}")
    actual = {
        path.name
        for path in candidate.iterdir()
        if path.is_file() and path.name != "candidate_manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("manifest file population mismatch")
    return manifest


def validate_external_ids(candidate: Path) -> None:
    entities = {
        row["entity_id"]
        for row in read_csv(PROJECT_ROOT / "data/entities.csv")
    }
    sources = {
        row["source_id"]
        for row in read_csv(PROJECT_ROOT / "data/source_register.csv")
    }
    canonical_claims = {
        row["claim_id"] for row in read_csv(PROJECT_ROOT / "data/claims.csv")
    }
    candidate_claims = index(
        read_csv(candidate / "candidate_claims.csv"),
        "claim_id",
        "candidate claims",
    )
    if canonical_claims & set(candidate_claims):
        raise ValueError("candidate claim ID collides with canonical claim")

    entity_tables = [
        "private_investment_earnings.csv",
        "private_investment_carrying_value_rollforwards.csv",
        "private_investment_earnings_bridges.csv",
        "private_investment_population_observations.csv",
    ]
    for file_name in entity_tables:
        for row in read_csv(candidate / file_name):
            if row["investor_entity_id"] not in entities:
                raise ValueError(
                    f"{file_name}: unknown investor {row['investor_entity_id']}"
                )
            investee = row.get("investee_entity_id", "")
            if investee and investee not in entities:
                raise ValueError(f"{file_name}: unknown investee {investee}")
            source_id = row.get("source_id", "")
            if source_id and source_id not in sources:
                raise ValueError(f"{file_name}: unknown source {source_id}")
            source_claim = row.get("source_claim_id", "")
            if source_claim and source_claim not in canonical_claims:
                raise ValueError(
                    f"{file_name}: unknown source claim {source_claim}"
                )
    for row in read_csv(candidate / "candidate_evidence_links.csv"):
        if row["claim_id"] not in candidate_claims:
            raise ValueError("candidate evidence link references unknown claim")
        if row["source_id"] not in sources:
            raise ValueError("candidate evidence link references unknown source")


def validate_semantics(candidate: Path) -> dict[str, int]:
    methods = index(
        read_csv(candidate / "private_investment_accounting_methods.csv"),
        "method_id",
        "methods",
    )
    earnings_rows = read_csv(candidate / "private_investment_earnings.csv")
    earnings = index(
        earnings_rows, "private_investment_earnings_id", "earnings"
    )
    if len(methods) != 11 or len(earnings) != 8:
        raise ValueError("method or earnings population changed")
    for row in earnings_rows:
        method_id = row["accounting_method_id"]
        if method_id and method_id not in methods:
            raise ValueError(f"unknown method {method_id}")
        if row["reporting_period_date_status"].startswith("unmapped"):
            raise ValueError("unmapped earnings period")
        if row["integration_status"] != "accepted_bounded_source_archive_hold":
            raise ValueError("earnings row lost source-archive hold")
        if row["cash_received_amount"] and row["cash_received_state"] != (
            "cash_proceeds_observed"
        ):
            raise ValueError("numeric cash proceeds lack observed state")
    if earnings["PIE-0003"]["additive_eligible"] != "0":
        raise ValueError("Amazon analytical sum is incorrectly additive")
    require_close(
        float(earnings["PIE-0001"]["pretax_gain_loss_amount"])
        + float(earnings["PIE-0002"]["pretax_gain_loss_amount"]),
        float(earnings["PIE-0003"]["pretax_gain_loss_amount"]),
        "Amazon component sum",
        1e-9,
    )
    for row in earnings_rows:
        numerator = number(row["pretax_gain_loss_amount"])
        denominator = number(row["reported_pretax_income_amount"])
        ratio = number(row["gain_share_of_pretax_income_pct"])
        if numerator is not None and denominator not in (None, 0) and ratio is not None:
            require_close(
                numerator / denominator * 100,
                ratio,
                f"{row['private_investment_earnings_id']} ratio",
                0.00006,
            )

    component_rows = read_csv(
        candidate / "private_investment_earnings_components.csv"
    )
    if {
        (row["parent_earnings_id"], row["component_earnings_id"])
        for row in component_rows
    } != {("PIE-0003", "PIE-0001"), ("PIE-0003", "PIE-0002")}:
        raise ValueError("Amazon analytical component junction changed")

    rollforward_rows = read_csv(
        candidate / "private_investment_carrying_value_rollforwards.csv"
    )
    rollforwards = index(rollforward_rows, "rollforward_id", "rollforwards")
    components = read_csv(
        candidate / "private_investment_rollforward_components.csv"
    )
    components_by_parent: dict[str, list[dict[str, str]]] = {}
    for row in components:
        if row["rollforward_id"] not in rollforwards:
            raise ValueError("orphan rollforward component")
        components_by_parent.setdefault(row["rollforward_id"], []).append(row)
    for rollforward_id, row in rollforwards.items():
        beginning = float(row["beginning_carrying_value_amount"])
        ending = float(row["ending_carrying_value_amount"])
        movement = sum(
            float(component["amount"])
            for component in components_by_parent[rollforward_id]
        )
        require_close(
            beginning + movement,
            ending,
            f"{rollforward_id} rollforward",
            1e-9,
        )
    aoci = rollforwards["PIR-0003"]
    if (
        aoci["instrument_scope"] != "AOCI_net_of_tax"
        or aoci["reconciliation_state"]
        != "exact_company_AOCI_rollforward_net_of_tax"
    ):
        raise ValueError("Amazon AOCI boundary changed")

    bridges = index(
        read_csv(candidate / "private_investment_earnings_bridges.csv"),
        "bridge_id",
        "bridges",
    )
    for bridge_id, row in bridges.items():
        require_close(
            float(row["reported_pretax_income_amount"])
            - float(row["identified_private_investment_gain_amount"]),
            float(row["arithmetic_pretax_residual_amount"]),
            f"{bridge_id} arithmetic residual",
            1e-9,
        )
        if "adjusted" not in row["analyst_derived_state"]:
            raise ValueError("bridge does not prohibit adjusted-earnings label")
    if (
        bridges["EQB-0001"]["cash_flow_reversal_amount"] != "15632"
        or bridges["EQB-0001"]["cash_flow_reversal_perimeter"]
        != "aggregate_nonoperating_income_reversal_not_component_specific"
    ):
        raise ValueError("Amazon cash-flow reversal perimeter was not split")

    population_rows = read_csv(
        candidate / "private_investment_population_observations.csv"
    )
    population = index(
        population_rows, "population_observation_id", "population"
    )
    if len(population) != 10:
        raise ValueError("population must contain ten accepted observations")
    required_classes = {
        "realized_gain_and_full_disposal",
        "material_carrying_exposure_no_identified_material_private_mark",
        "material_carrying_exposure_no_isolated_period_private_gain",
        "mixed_portfolio_context_not_private_attributable",
        "new_monitored_private_position",
    }
    if not required_classes <= {
        row["observation_class"] for row in population_rows
    }:
        raise ValueError("population lacks realized or negative/monitor cases")

    claims = index(
        read_csv(candidate / "candidate_claims.csv"),
        "claim_id",
        "candidate claims",
    )
    if set(claims) != {
        "CLM-070035",
        "CLM-070036",
        "CLM-070037",
        "CLM-070038",
        "CLM-110003",
    }:
        raise ValueError("candidate claim identity set changed")
    if any(
        row["review_status"] != "accepted_bounded_source_archive_hold"
        for row in claims.values()
    ):
        raise ValueError("candidate claims lost source-archive hold")

    handoffs = read_csv(candidate / "candidate_handoffs.csv")
    if any(row["readiness_status"] == "ready" for row in handoffs):
        raise ValueError("handoff incorrectly cleared retained external holds")
    if any(
        forbidden in " ".join(row.values()).lower()
        for row in handoffs
        for forbidden in ("top-20 total established", "s&p total established")
    ):
        raise ValueError("blocked population claim entered a handoff")

    audit = read_csv(candidate / "transformation_audit.csv")
    if len(audit) != 500:
        raise ValueError("field-level transformation audit population changed")
    if any(row["lossless_status"] != "preserved_with_source_value" for row in audit):
        raise ValueError("field-level transformation audit contains field loss")
    return {
        "methods": len(methods),
        "earnings": len(earnings),
        "rollforwards": len(rollforwards),
        "rollforward_components": len(components),
        "bridges": len(bridges),
        "population_observations": len(population),
        "candidate_claims": len(claims),
        "candidate_handoffs": len(handoffs),
        "transformation_audit_rows": len(audit),
    }


def validate_sqlite_load(candidate: Path) -> dict[str, int]:
    connection = sqlite3.connect(":memory:")
    connection.executescript((candidate / "candidate_schema.sql").read_text())
    tables = [
        "private_investment_accounting_methods",
        "private_investment_earnings",
        "private_investment_earnings_components",
        "private_investment_carrying_value_rollforwards",
        "private_investment_rollforward_components",
        "private_investment_earnings_bridges",
        "private_investment_bridge_components",
        "private_investment_population_observations",
    ]
    counts: dict[str, int] = {}
    for table in tables:
        rows = read_csv(candidate / f"{table}.csv")
        columns = list(rows[0])
        table_info = {
            row[1]: {"notnull": bool(row[3]), "primary_key": bool(row[5])}
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        placeholders = ",".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
            [
                tuple(
                    (
                        ""
                        if row[column] == ""
                        and (
                            table_info[column]["notnull"]
                            or table_info[column]["primary_key"]
                        )
                        else None
                        if row[column] == ""
                        else row[column]
                    )
                    for column in columns
                )
                for row in rows
            ],
        )
        counts[table] = len(rows)
    foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if foreign_keys:
        raise ValueError(f"candidate foreign-key failures: {foreign_keys}")
    if integrity != "ok":
        raise ValueError(f"candidate integrity failure: {integrity}")
    return counts


def validate_determinism(candidate: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="pie-candidate-a-") as first_tmp:
        with tempfile.TemporaryDirectory(prefix="pie-candidate-b-") as second_tmp:
            first = Path(first_tmp) / "candidate"
            second = Path(second_tmp) / "candidate"
            for output in (first, second):
                subprocess.run(
                    [
                        str(REPO_ROOT / ".venv/bin/python"),
                        str(BUILDER),
                        "--output",
                        str(output),
                    ],
                    check=True,
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
            names = {path.name for path in first.iterdir()}
            if names != {path.name for path in second.iterdir()}:
                raise ValueError("clean candidate build file sets differ")
            if names != {path.name for path in candidate.iterdir()}:
                raise ValueError("clean build file set differs from sealed candidate")
            for name in names:
                first_bytes = (first / name).read_bytes()
                if first_bytes != (second / name).read_bytes():
                    raise ValueError(f"clean candidate builds differ: {name}")
                if first_bytes != (candidate / name).read_bytes():
                    raise ValueError(f"sealed candidate differs from rebuild: {name}")
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    manifest = validate_manifest(candidate)
    validate_external_ids(candidate)
    semantic_counts = validate_semantics(candidate)
    sqlite_counts = validate_sqlite_load(candidate)
    deterministic_files = validate_determinism(candidate)
    results = {
        "status": "PASS",
        "candidate_status": manifest["status"],
        "canonical_write": "NOT_AUTHORIZED_PENDING_FULL_SECOND_HALF_INTEGRATION",
        "review_verdict": manifest["review_verdict"],
        "retained_holds": manifest["retained_holds"],
        "semantic_counts": semantic_counts,
        "sqlite_loaded_rows": sqlite_counts,
        "sqlite_integrity": "ok",
        "sqlite_foreign_key_failures": 0,
        "clean_builds_compared": 2,
        "deterministic_files_compared": deterministic_files,
        "candidate_manifest_sha256": sha256(
            candidate / "candidate_manifest.json"
        ),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
