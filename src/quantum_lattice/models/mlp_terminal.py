"""Training progress display sharing the sampler's terminal lifecycle controls."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
import shutil
from typing import Any

from quantum_lattice.simulation.terminal import RunConsole, _clean, _duration


_TRAINING_CONTEXT = ContextVar("mlp_training_display", default={})


@contextmanager
def training_context(**context):
    """Attach queue labels to the display without changing learning settings.

    Nested contexts merge labels, so a pair can supply its mode while retaining
    a queue's size/progress. All values are display-only and are never saved in
    training checkpoints or used by the optimizer.
    """
    merged = {**_TRAINING_CONTEXT.get(), **context}
    merged.setdefault("stop_requested", {"requested": False})
    token = _TRAINING_CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _TRAINING_CONTEXT.reset(token)


def training_stop_requested() -> bool:
    """Whether q/Ctrl+C requested that the surrounding sequence stop."""
    state = _TRAINING_CONTEXT.get().get("stop_requested", {})
    return bool(state.get("requested", False))


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
        self.training_context = dict(_TRAINING_CONTEXT.get())
        super().__init__(max_steps, title="Paper-style MLP training",
                         interactive=interactive, **kwargs)

    def _request_sequence_stop(self) -> None:
        state = self.training_context.get("stop_requested")
        if isinstance(state, dict):
            state["requested"] = True
        callback = self.training_context.get("request_stop")
        if callable(callback):
            callback()

    def poll(self) -> str | None:
        try:
            command = super().poll()
        except KeyboardInterrupt:
            self._request_sequence_stop()
            raise
        if command == "quit":
            self._request_sequence_stop()
        return command

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # A key can arrive during the final optimizer chunk, after the engine's
        # last poll. Retain that request before restoring the terminal so the
        # queue cannot accidentally launch another model. Finalizing the model
        # remains the runner's responsibility; this is only a sequence flag.
        try:
            if exc_type is KeyboardInterrupt:
                self._request_sequence_stop()
            pending_pause = False
            for _ in range(64):
                try:
                    command = self.poll()
                except KeyboardInterrupt:
                    break
                if command == "pause":
                    pending_pause = True
                elif command == "resume":
                    pending_pause = False
                if command is None or command == "quit":
                    break
            if pending_pause:
                # The model may already be complete. Preserve the pause at
                # the queue boundary rather than silently starting its successor.
                self._request_sequence_stop()
        finally:
            super().__exit__(exc_type, exc_value, traceback)

    def render(self, status: dict) -> None:
        """Render training/evaluation state; omit sampling-specific counters."""
        step = int(status.get("step", status.get("updates", 0)))
        maximum = int(status.get("max_steps", self.max_steps))
        epoch = int(status.get("epoch", 0))
        state_name = _clean(status.get("state", "training")).lower()
        mode = state_name.upper()
        finished = state_name in {"trained", "evaluated"} or bool(status.get("stop_reason"))
        message = _clean(status.get("message", ""))
        state = (state_name, finished, step >= maximum if maximum > 0 else False)
        now = self._clock()
        interval = 0.3 if self.interactive else 5.0
        if (self._last_render is not None and now - self._last_render < interval
                and self._last_state == state):
            return
        self._last_render = now
        self._last_state = state

        fraction = min(1.0, max(0.0, step / maximum)) if maximum > 0 else 0.0
        elapsed = float(status.get("elapsed_seconds", 0.0))
        if finished:
            eta = 0.0
        elif "eta_seconds" in status:
            eta = math.inf if status["eta_seconds"] is None else float(status["eta_seconds"])
        else:
            eta = elapsed * (1 - fraction) / fraction if fraction > 0 else math.inf
        train_mse = _metric(status.get("train_mse"))
        validation_mse = _metric(status.get("validation_mse"))
        best_mse = _metric(status.get("best_validation_mse"))
        best_step = status.get("best_step")
        best_at = "--" if best_step is None else str(int(best_step))
        stopper = status.get("early_stopping", {})
        policy = (f"early stop checks {status.get('early_stopping_wait', 0)}/{stopper.get('patience', '?')}"
                  if stopper.get("enabled") else "fixed update budget")
        if status.get("stop_reason"):
            policy += f"; stop reason {status['stop_reason']}"
        eta_label = "ETA to ceiling" if stopper.get("enabled") and not finished else "ETA"

        labels = []
        if status.get("device"):
            labels.append(_clean(status["device"]))
        if self.training_context.get("size") is not None:
            labels.append(f"L={self.training_context['size']}")
        if self.training_context.get("mode"):
            labels.append(_clean(self.training_context["mode"]))
        if self.training_context.get("queue_index") is not None:
            labels.append(f"model {self.training_context['queue_index']}/{self.training_context.get('queue_total', '?')}")
        context_label = " | ".join(labels)

        if not self.interactive:
            self.output_stream.write(
                (f"{context_label} | " if context_label else "")
                + f"[{mode}] Updates {step:,}/{maximum:,}; epoch {epoch}; "
                f"train MSE {train_mse}; validation MSE {validation_mse}; "
                f"best validation MSE {best_mse} at update {best_at}; "
                f"elapsed {_duration(elapsed)}; {eta_label} ~{_duration(eta)}; {policy}"
                + (f"; {message}" if message else "") + "\n"
            )
            self.output_stream.flush()
            return

        width = max(20, min(100, shutil.get_terminal_size(fallback=(80, 24)).columns - 1))
        bar_width = max(5, min(16, width - 58))
        filled = int(fraction * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        compact_policy = (f"early stop {status.get('early_stopping_wait', 0)}/{stopper.get('patience', '?')}"
                          if stopper.get("enabled") else "fixed budget")
        lines = [
            "MLP training" + (f" | {context_label}" if context_label else ""),
            f"{mode} [{bar}] {100 * fraction:4.1f}% | {step:,}/{maximum:,} updates | epoch {epoch}",
            f"MSE train {train_mse} | validation {validation_mse} | best {best_mse} @{best_at}",
            f"Elapsed {_duration(elapsed)} | {eta_label} ~{_duration(eta)} | {compact_policy}",
        ]
        if message or status.get("stop_reason"):
            lines[-1] = message or f"Stopped: {status['stop_reason']} | elapsed {_duration(elapsed)}"
        lines.append(_clean(status.get("controls", "p pause | r resume | q save and quit | Ctrl+C save and quit")))
        if self._line_count:
            self.output_stream.write(f"\x1b[{self._line_count}A")
        self.output_stream.write("".join("\r\x1b[2K" + line[:width] + "\n" for line in lines))
        self.output_stream.flush()
        self._line_count = len(lines)
