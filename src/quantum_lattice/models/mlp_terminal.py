"""Training progress display sharing the sampler's terminal lifecycle controls."""

from __future__ import annotations

import math
import shutil
from typing import Any

from quantum_lattice.simulation.terminal import RunConsole, _clean, _duration


def _metric(value: Any) -> str:
    if value is None:
        return "--"
    number = float(value)
    return f"{number:.6g}" if math.isfinite(number) else "--"


class TrainingConsole(RunConsole):
    """Display optimizer updates and validation selection, with p/r/q controls.

    The parent supplies terminal restoration and nonblocking command polling.
    The training runner owns checkpointing and determines when a pause is safe.
    """

    def __init__(self, max_steps: int, interactive: bool = True, **kwargs: Any) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        super().__init__(max_steps, title="Paper-style MLP training",
                         interactive=interactive, **kwargs)

    def render(self, status: dict) -> None:
        """Render training/evaluation state; omit sampling-specific counters."""
        step = int(status.get("step", status.get("updates", 0)))
        maximum = int(status.get("max_steps", self.max_steps))
        epoch = int(status.get("epoch", 0))
        state_name = _clean(status.get("state", "training")).lower()
        mode = state_name.upper()
        message = _clean(status.get("message", ""))
        state = (state_name, step >= maximum if maximum > 0 else False)
        now = self._clock()
        interval = 0.3 if self.interactive else 5.0
        if (self._last_render is not None and now - self._last_render < interval
                and self._last_state == state):
            return
        self._last_render = now
        self._last_state = state

        fraction = min(1.0, max(0.0, step / maximum)) if maximum > 0 else 0.0
        elapsed = float(status.get("elapsed_seconds", 0.0))
        if "eta_seconds" in status:
            eta = math.inf if status["eta_seconds"] is None else float(status["eta_seconds"])
        else:
            eta = elapsed * (1 - fraction) / fraction if fraction > 0 else math.inf
        train_mse = _metric(status.get("train_mse"))
        validation_mse = _metric(status.get("validation_mse"))
        best_mse = _metric(status.get("best_validation_mse"))
        best_step = status.get("best_step")
        best_at = "--" if best_step is None else str(int(best_step))

        if not self.interactive:
            self.output_stream.write(
                f"[{mode}] Updates {step:,}/{maximum:,}; epoch {epoch}; "
                f"train MSE {train_mse}; validation MSE {validation_mse}; "
                f"best validation MSE {best_mse} at update {best_at}; "
                f"elapsed {_duration(elapsed)}; ETA ~{_duration(eta)}"
                + (f"; {message}" if message else "") + "\n"
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
            f"Optimizer updates {step:,}/{maximum:,}  |  epoch {epoch}",
            f"Train MSE {train_mse}  |  validation MSE {validation_mse}",
            f"Best validation MSE {best_mse}  |  selected update {best_at}",
            f"Elapsed {_duration(elapsed)}  |  ETA ~{_duration(eta)}",
            message or "Model selection uses validation data only.",
            _clean(status.get("controls", "p pause  |  r resume  |  q save and quit  |  Ctrl+C save and quit")),
        ]
        if self._line_count:
            self.output_stream.write(f"\x1b[{self._line_count}A")
        self.output_stream.write("".join("\r\x1b[2K" + line[:width] + "\n" for line in lines))
        self.output_stream.flush()
        self._line_count = len(lines)
