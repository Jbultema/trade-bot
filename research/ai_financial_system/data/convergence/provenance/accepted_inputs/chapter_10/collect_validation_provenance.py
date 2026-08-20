#!/usr/bin/env python3
"""Collect deterministic dedicated-validation and Ruff provenance."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    validator_relative = (
        HERE / "validate_packet.py"
    ).relative_to(ROOT).as_posix()
    validation = run(
        [
            sys.executable,
            validator_relative,
        ]
    )
    validation_payload: dict[str, Any] = (
        json.loads(validation.stdout)
        if validation.stdout.strip()
        else {"status": "NO_OUTPUT"}
    )

    python_paths = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(HERE.glob("*.py"))
    ]
    version = run([sys.executable, "-m", "ruff", "--version"])
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--output-format",
        "json",
        *python_paths,
    ]
    ruff = run(command)
    diagnostics: list[dict[str, Any]] | dict[str, Any]
    try:
        diagnostics = json.loads(ruff.stdout) if ruff.stdout.strip() else []
    except json.JSONDecodeError:
        diagnostics = {"unparsed_stdout": ruff.stdout}
    payload = {
        "status": "PASS" if ruff.returncode == 0 else "FAIL",
        "tool": "ruff",
        "tool_version": version.stdout.strip(),
        "command": [
            "python",
            "-m",
            "ruff",
            "check",
            "--output-format",
            "json",
            *python_paths,
        ],
        "returncode": ruff.returncode,
        "diagnostic_count": (
            len(diagnostics) if isinstance(diagnostics, list) else None
        ),
        "diagnostics": diagnostics,
        "stderr": ruff.stderr.strip(),
        "scope": python_paths,
        "boundary": "static analysis provenance only",
    }
    (HERE / "ruff_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    combined_pass = (
        validation.returncode == 0
        and validation_payload.get("status") == "PASS"
        and ruff.returncode == 0
    )
    print(
        json.dumps(
            {
                "status": "PASS" if combined_pass else "FAIL",
                "dedicated_validation": validation_payload.get("status"),
                "ruff": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if combined_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
