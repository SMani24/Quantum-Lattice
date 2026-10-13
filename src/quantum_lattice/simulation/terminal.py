"""Small terminal progress panel; simulation control stays with the runner.

Interactive terminals accept ``p``, ``r``, and ``q`` without Enter. Redirected
output receives occasional plain progress lines, and never terminal escapes.
"""

from __future__ import annotations

from collections import deque
import math
import os
import select
import shutil
import sys
import time
from typing import Callable, TextIO


def _duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _clean(text: object) -> str:
    """Keep messages on one line and prevent embedded terminal controls."""
    return " ".join("".join(c for c in str(text) if c >= " " and c != "\x7f").split())


class RunConsole:
    """Context-managed progress display and nonblocking keyboard input.

    A call to :meth:`poll` returns one of ``pause``, ``resume``, ``quit`` or
    ``None``. The caller decides when a pause/checkpoint is safe. In particular,
    requesting a pause does not make the panel claim that workers have paused.
    ``Ctrl+C`` is left enabled so the runner can checkpoint on KeyboardInterrupt.
    """

    def __init__(
        self,
        total_jobs: int,
        title: str = "Ground-state sample generation",
        interactive: bool = True,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total_jobs = total_jobs
        self.title = _clean(title)
        self.input_stream = input_stream if input_stream is not None else sys.stdin
        self.output_stream = output_stream if output_stream is not None else sys.stdout
        self._clock = clock
        self._interactive_requested = interactive
        self.interactive = False
        self._fd: int | None = None
        self._terminal_settings: list | None = None
        self._commands: deque[str] = deque()
        self._last_render: float | None = None
        self._last_state: tuple | None = None
        self._line_count = 0
        self._entered = False

    def __enter__(self) -> RunConsole:
        if self._entered:
            raise RuntimeError("RunConsole is already active")
        self._entered = True
        try:
            if (
                self._interactive_requested
                and os.name == "posix"
                and os.environ.get("TERM") != "dumb"
                and self.input_stream.isatty()
                and self.output_stream.isatty()
            ):
                import termios
                import tty

                self._fd = self.input_stream.fileno()
                self._terminal_settings = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self.interactive = True
                self.output_stream.write("\x1b[?25l")
                self.output_stream.flush()
        except (AttributeError, OSError, ValueError):
            # A pseudo-terminal can disappear while a job is being started.
            # Restore any changed settings and continue with the plain display.
            self._restore_terminal()
        return self

    def _restore_terminal(self) -> None:
        if self._terminal_settings is not None and self._fd is not None:
            import termios

            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._terminal_settings)
            except (OSError, termios.error):
                pass
        self._terminal_settings = None
        self._fd = None
        self.interactive = False

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        was_interactive = self.interactive
        self._restore_terminal()
        self._entered = False
        if was_interactive:
            try:
                self.output_stream.write("\x1b[?25h")
                self.output_stream.flush()
            except OSError:
                pass

    def poll(self) -> str | None:
        """Read available keystrokes without blocking simulation progress."""
        if self._commands:
            return self._commands.popleft()
        if not self.interactive or self._fd is None:
            return None
        try:
            ready, _, _ = select.select([self._fd], [], [], 0)
            if not ready:
                return None
            data = os.read(self._fd, 64)
            if not data:
                return None
        except (OSError, ValueError):
            return None
        commands = {"p": "pause", "r": "resume", "q": "quit"}
        for key in data.decode("ascii", errors="ignore").lower():
            if key == "\x03":
                raise KeyboardInterrupt
            if key in commands:
                self._commands.append(commands[key])
        return self._commands.popleft() if self._commands else None

    def render(self, status: dict) -> None:
        """Render the latest snapshot, with automatic redraw throttling.

        Expected keys are completed, total, running, paused, elapsed_seconds,
        attempts_done, attempts_total, samples_done, samples_total, workers and
        message. Missing counters default to zero. Pause/resume and completion
        bypass throttling, so control changes remain visible immediately.
        ``pausing`` reports a request that workers have not finished saving.
        An optional ``eta_seconds`` overrides the estimate; None means unknown.
        """
        completed = int(status.get("completed", 0))
        total = int(status.get("total", self.total_jobs))
        paused = bool(status.get("paused", False))
        pausing = bool(status.get("pausing", False))
        phase = status.get("phase", "sampling")
        message = _clean(status.get("message", ""))
        state = (paused, pausing, phase, completed >= total if total > 0 else False)
        now = self._clock()
        interval = 0.3 if self.interactive else 5.0
        if (
            self._last_render is not None
            and now - self._last_render < interval
            and self._last_state == state
        ):
            return
        self._last_render = now
        self._last_state = state

        attempts_done = int(status.get("attempts_done", 0))
        attempts_total = int(status.get("attempts_total", 0))
        samples_done = int(status.get("samples_done", 0))
        samples_total = int(status.get("samples_total", 0))
        elapsed = float(status.get("elapsed_seconds", 0.0))
        if attempts_total > 0:
            fraction = min(1.0, max(0.0, attempts_done / attempts_total))
        else:
            fraction = min(1.0, max(0.0, completed / total)) if total else 0.0
        if "eta_seconds" in status:
            override = status["eta_seconds"]
            eta = math.inf if override is None else float(override)
        else:
            eta = elapsed * (1 - fraction) / fraction if fraction > 0 else math.inf
        mode = "PAUSED" if paused else "PAUSING" if pausing else "VERIFYING" if phase == "verifying" else "RUNNING"
        if total > 0 and completed >= total:
            mode = "COMPLETE"
        running = status.get("running", 0)
        workers = status.get("workers", 1)
        if not self.interactive:
            if phase == "verifying":
                self.output_stream.write(
                    f"[{mode}] Hamiltonians checked {completed}/{total}; "
                    f"elapsed {_duration(elapsed)}; ETA ~{_duration(eta)}; {message}\n"
                )
                self.output_stream.flush()
                return
            self.output_stream.write(
                f"[{mode}] Hamiltonians {completed}/{total}; "
                f"attempts {attempts_done:,}/{attempts_total:,}; "
                f"samples {samples_done:,}/{samples_total:,}; "
                f"elapsed {_duration(elapsed)}; ETA ~{_duration(eta)}"
                + (f"; {message}" if message else "")
                + "\n"
            )
            self.output_stream.flush()
            return

        width = max(20, min(100, shutil.get_terminal_size(fallback=(80, 24)).columns - 1))
        bar_width = max(5, min(36, width - 14))
        filled = int(fraction * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        lines = [
            self.title,
            f"{mode}  [{bar}] {100 * fraction:5.1f}%",
            f"Hamiltonians {completed}/{total}  |  active {running}  |  workers {workers}",
            f"Attempts {attempts_done:,}/{attempts_total:,}",
            f"Retained samples {samples_done:,}/{samples_total:,}",
            f"Elapsed {_duration(elapsed)}  |  ETA ~{_duration(eta)}",
            message or "Working; checkpoints are saved at chunk boundaries.",
            _clean(status.get("controls", "p pause  |  r resume  |  q save and quit  |  Ctrl+C save and quit")),
        ]
        if self._line_count:
            self.output_stream.write(f"\x1b[{self._line_count}A")
        self.output_stream.write(
            "".join("\r\x1b[2K" + line[:width] + "\n" for line in lines)
        )
        self.output_stream.flush()
        self._line_count = len(lines)
