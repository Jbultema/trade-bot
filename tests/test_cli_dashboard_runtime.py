from __future__ import annotations

import signal
import subprocess
from pathlib import Path

from typer.testing import CliRunner

import trade_bot.cli as cli_module
from trade_bot.cli import app


def test_run_dashboard_starts_managed_streamlit_process(monkeypatch, tmp_path: Path) -> None:
    popen_calls: list[tuple[list[str], dict[str, object]]] = []

    class FakePopen:
        pid = 12345

        def __init__(self, command: list[str], **kwargs: object) -> None:
            popen_calls.append((command, kwargs))

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(cli_module, "_process_exists", lambda pid: False)
    pid_path = tmp_path / "streamlit.pid"
    log_path = tmp_path / "streamlit.log"

    result = CliRunner().invoke(
        app,
        [
            "run-dashboard",
            "--pid-path",
            str(pid_path),
            "--log-path",
            str(log_path),
            "--port",
            "8765",
        ],
    )

    assert result.exit_code == 0, result.output
    assert pid_path.read_text(encoding="utf-8").strip() == "12345"
    command = popen_calls[0][0]
    assert command[:3] == [cli_module.sys.executable, "-m", "streamlit"]
    assert "--server.fileWatcherType" in command
    assert "none" in command
    assert "--server.port" in command
    assert "8765" in command
    assert popen_calls[0][1]["start_new_session"] is True


def test_stop_dashboard_escalates_when_graceful_shutdown_hangs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "streamlit.pid"
    pid_path.write_text("12345\n", encoding="utf-8")
    signals: list[int] = []

    def fake_signal_process(pid: int, sig: int) -> None:
        assert pid == 12345
        signals.append(sig)

    def fake_process_exists(pid: int) -> bool:
        assert pid == 12345
        return signal.SIGKILL not in signals

    monkeypatch.setattr(cli_module, "_signal_process", fake_signal_process)
    monkeypatch.setattr(cli_module, "_process_exists", fake_process_exists)
    monkeypatch.setattr(cli_module.time, "sleep", lambda seconds: None)

    result = CliRunner().invoke(
        app,
        [
            "stop-dashboard",
            "--pid-path",
            str(pid_path),
            "--timeout-seconds",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert not pid_path.exists()
