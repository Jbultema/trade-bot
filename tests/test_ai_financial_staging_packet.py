from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "research/ai_financial_system/scripts/validate_staging_packet.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_staging_packet_test_module",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, str]],
    *,
    declared_key: list[str] | None,
) -> object:
    validator = _load_script()
    monkeypatch.setattr(validator, "load_source_urls", lambda: {})
    path = tmp_path / "junction.csv"
    _write_csv(path, rows)
    entry: dict[str, object] = {"path": "junction.csv"}
    if declared_key is not None:
        entry["primary_key_columns"] = declared_key
    result = validator.Validation(lane=tmp_path.name)

    validator.validate_csv_links(tmp_path, result, {"files": [entry]})

    return result


def test_declared_composite_key_allows_repeated_parent_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _validate_rows(
        tmp_path,
        monkeypatch,
        [
            {"claim_id": "CLM-1", "source_id": "SRC-1"},
            {"claim_id": "CLM-1", "source_id": "SRC-2"},
        ],
        declared_key=["claim_id", "source_id"],
    )

    assert result.passed


def test_declared_composite_key_rejects_true_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _validate_rows(
        tmp_path,
        monkeypatch,
        [
            {"claim_id": "CLM-1", "source_id": "SRC-1"},
            {"claim_id": "CLM-1", "source_id": "SRC-1"},
        ],
        declared_key=["claim_id", "source_id"],
    )

    assert not result.passed
    assert any("duplicate provisional ID CLM-1|SRC-1" in error for error in result.errors)


def test_declared_composite_key_rejects_blank_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _validate_rows(
        tmp_path,
        monkeypatch,
        [{"claim_id": "CLM-1", "source_id": ""}],
        declared_key=["claim_id", "source_id"],
    )

    assert not result.passed
    assert any("blank primary-key component" in error for error in result.errors)


def test_declared_key_rejects_column_absent_from_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _validate_rows(
        tmp_path,
        monkeypatch,
        [{"claim_id": "CLM-1", "source_id": "SRC-1"}],
        declared_key=["claim_id", "locator"],
    )

    assert not result.passed
    assert any(
        "manifest primary_key_columns are absent from CSV: locator" in error
        for error in result.errors
    )


def test_legacy_key_inference_still_rejects_duplicate_first_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _validate_rows(
        tmp_path,
        monkeypatch,
        [
            {"claim_id": "CLM-1", "source_id": "SRC-1"},
            {"claim_id": "CLM-1", "source_id": "SRC-2"},
        ],
        declared_key=None,
    )

    assert not result.passed
    assert any("duplicate provisional ID CLM-1" in error for error in result.errors)


def test_explicit_empty_key_disables_legacy_uniqueness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _validate_rows(
        tmp_path,
        monkeypatch,
        [
            {"claim_id": "CLM-1", "source_id": "SRC-1"},
            {"claim_id": "CLM-1", "source_id": "SRC-2"},
        ],
        declared_key=[],
    )

    assert result.passed
