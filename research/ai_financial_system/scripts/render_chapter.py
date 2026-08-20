from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", required=True)
    args = parser.parse_args()
    chapter = args.chapter.zfill(2)
    source = PROJECT_ROOT / f"chapters/chapter_{chapter}/chapter.html"
    output = (
        REPO_ROOT
        / f"reports/ai_financial_system/chapters/chapter_{chapter}"
        / f"AI_Financial_System_Chapter_{chapter}.docx"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "/usr/bin/textutil",
            "-convert",
            "docx",
            "-output",
            str(output),
            str(source),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"DOCX conversion did not produce a valid file: {output}")
    print(output)


if __name__ == "__main__":
    main()
