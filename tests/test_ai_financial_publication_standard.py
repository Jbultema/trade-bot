from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "research/ai_financial_system/scripts/audit_publication_standard.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_publication_standard_test_module",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_chapter_is_an_explicit_publication_failure(
    tmp_path: Path,
) -> None:
    auditor = _load_script()
    auditor.PROJECT_ROOT = tmp_path
    auditor.REPO_ROOT = tmp_path

    result = auditor.audit_chapter(16)

    assert result["chapter"] == 16
    assert result["heading_count"] == 0
    assert result["publication_gate"] == "missing_source"
    assert result["observed_facts"] == "source_missing"


def test_cli_exits_nonzero_when_any_chapter_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_script()
    auditor.PROJECT_ROOT = tmp_path
    auditor.REPO_ROOT = tmp_path
    output = tmp_path / "audit.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_publication_standard.py",
            "--through",
            "2",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        auditor.main()

    assert exc_info.value.code == 1
    assert output.is_file()
    assert "missing_source" in output.read_text(encoding="utf-8")
