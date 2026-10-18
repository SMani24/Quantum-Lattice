"""Small, injectable terminal menu for the observable-map campaign."""
from __future__ import annotations

from collections.abc import Callable


def dashboard(run, *, status: Callable, launch: Callable,
              read: Callable = input, write: Callable = print) -> int:
    """A status view and three run choices; Ctrl+C while idle exits cleanly.

    The launcher owns cooperative training pauses. On return, the menu remains
    available so the user can inspect progress or quit without losing state.
    """
    while True:
        write("\nObservable-map CNN comparison — GPU, one model at a time")
        write(status())
        write("1  Run / continue L=4\n2  Run / continue L=3\n"
              "3  Run / continue both\n4  Progress and results\nq  Quit")
        try:
            choice = read("Choose: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            write("")
            return 0
        if choice in ("q", "quit", "exit"):
            return 0
        if choice in ("1", "2", "3"):
            sizes = {"1": [4], "2": [3], "3": None}[choice]
            launch(sizes=sizes)
        elif choice == "4":
            write(status())
            write(f"Results: {run}/RESULTS.md")
        else:
            write("Choose 1, 2, 3, 4 or q.")
