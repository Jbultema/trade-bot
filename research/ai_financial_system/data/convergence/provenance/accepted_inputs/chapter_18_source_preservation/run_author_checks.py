#!/usr/bin/env python3
"""Run Ruff and generic staging validation for the corrected author packet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AI_ROOT = HERE.parents[1]
REPO = HERE.parents[3]
GENERIC = AI_ROOT / "scripts/validate_staging_packet.py"


def main() -> int:
    ruff = subprocess.run(
        [
            str(REPO / ".venv/bin/ruff"),
            "check",
            *[str(path) for path in sorted(HERE.glob("*.py"))],
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    ruff_result = {
        "status": "PASS" if ruff.returncode == 0 else "FAIL",
        "returncode": ruff.returncode,
        "stdout": ruff.stdout,
        "stderr": ruff.stderr,
    }
    (HERE / "ruff_results.json").write_text(
        json.dumps(ruff_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generic = subprocess.run(
        [sys.executable, str(GENERIC), str(HERE), "--json"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    generic_result = (
        json.loads(generic.stdout)
        if generic.returncode == 0
        else {
            "passed": False,
            "returncode": generic.returncode,
            "stdout": generic.stdout,
            "stderr": generic.stderr,
        }
    )
    (HERE / "generic_validation_results.json").write_text(
        json.dumps(generic_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "ruff": ruff_result["status"],
        "generic": "PASS" if generic_result.get("passed") else "FAIL",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(value == "PASS" for value in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
