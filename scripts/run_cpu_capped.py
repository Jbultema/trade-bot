#!/usr/bin/env python3
"""Run a command with a conservative single-core CPU duty-cycle cap.

This wrapper is intended for local trade-bot research jobs. It forces common
numeric runtimes to one thread and pauses the entire child process group during
the off portion of each duty-cycle period.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence

THREAD_LIMIT_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
MAX_DUTY_CYCLE = 0.70


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one command with numeric libraries restricted to one thread "
            "and the process group capped by a run/pause duty cycle."
        )
    )
    parser.add_argument(
        "--duty-cycle",
        type=float,
        default=0.65,
        help="Running share of each period; maximum allowed is 0.70.",
    )
    parser.add_argument(
        "--period-seconds",
        type=float,
        default=1.0,
        help="Duty-cycle period in seconds.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not 0.05 <= args.duty_cycle <= MAX_DUTY_CYCLE:
        parser.error(f"--duty-cycle must be between 0.05 and {MAX_DUTY_CYCLE:.2f}")
    if not 0.20 <= args.period_seconds <= 10.0:
        parser.error("--period-seconds must be between 0.20 and 10.0")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def _signal_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if process.poll() is None:
        os.killpg(process.pid, sig)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    env = os.environ.copy()
    for name in THREAD_LIMIT_ENV:
        env[name] = "1"

    process = subprocess.Popen(
        args.command,
        env=env,
        start_new_session=True,
    )
    run_seconds = args.period_seconds * args.duty_cycle
    pause_seconds = args.period_seconds - run_seconds

    def forward_signal(signum: int, _frame: object) -> None:
        if process.poll() is None:
            _signal_group(process, signal.SIGCONT)
            _signal_group(process, signal.Signals(signum))

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    try:
        while process.poll() is None:
            time.sleep(run_seconds)
            if process.poll() is not None:
                break
            _signal_group(process, signal.SIGSTOP)
            time.sleep(pause_seconds)
            _signal_group(process, signal.SIGCONT)
    finally:
        if process.poll() is None:
            _signal_group(process, signal.SIGCONT)
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
