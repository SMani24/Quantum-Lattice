"""Saved MLP learning and held-out prediction figures for simulation data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np


_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 14, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9ca7b4", "axes.labelcolor": "#243247",
    "xtick.color": "#58677a", "ytick.color": "#58677a",
    "legend.frameon": False, "pdf.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
}
_BIN_EDGES = np.array([0., .1, .3, .6, 1., 1.4, 1.7])


def _save(figure, output_dir, stem):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for extension in ("png", "pdf"):
            path = directory / f"{stem}.{extension}"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            paths.append(path)
    finally:
        plt.close(figure)
    return paths


def plot_learning_history(history: list[dict], output_dir: Path) -> list[Path]:
    """Plot full training and validation MSE at the recorded optimizer steps."""
    if not history:
        raise ValueError("learning history must contain at least one validation event")
    steps = np.asarray([event["step"] for event in history], dtype=float)
    train = np.asarray([event["train_mse"] for event in history], dtype=float)
    validation = np.asarray([event["validation_mse"] for event in history], dtype=float)
    if not np.isfinite([steps, train, validation]).all() or np.any(steps < 0) or np.any(np.diff(steps) < 0):
        raise ValueError("history requires finite, nondecreasing optimizer steps and losses")
    if np.any(train < 0) or np.any(validation < 0):
        raise ValueError("mean squared losses must be nonnegative")
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(7.4, 4.9))
        axis.plot(steps, train, color="#3568b0", linewidth=1.8, label="Training")
        axis.plot(steps, validation, color="#d66d3b", linewidth=1.8, label="Validation")
        best = int(np.argmin(validation))
        axis.scatter([steps[best]], [validation[best]], s=38, color="#d66d3b", zorder=4,
                     label="Lowest validation MSE")
        axis.set(title="MLP training", xlabel="Optimizer step", ylabel="Mean squared error in $|b|$")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#e3e8ee", linewidth=.7)
        axis.legend()
        figure.text(.12, .025, "Known-field simulation estimators; checkpoint selection uses validation only.",
                    color="#58677a", fontsize=8)
        figure.subplots_adjust(bottom=.18)
        return _save(figure, output_dir, "learning_curve")


def _prediction_vectors(predictions, targets):
    prediction, target = np.asarray(predictions, dtype=float), np.asarray(targets, dtype=float)
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if prediction.ndim != 1 or prediction.shape != target.shape or not len(target):
        raise ValueError("predictions and targets must be matching nonempty vectors or (n,1) arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("raw predictions must be finite and targets finite nonnegative magnitudes")
    return prediction, target


def _binned_mae(prediction, target):
    result = []
    errors = np.abs(prediction - target)
    for index, (low, high) in enumerate(zip(_BIN_EDGES[:-1], _BIN_EDGES[1:])):
        last = index == len(_BIN_EDGES) - 2
        mask = (target >= low) & (target <= high if last else target < high)
        result.append({"low": low, "high": high, "count": int(np.sum(mask)),
                       "mae": float(np.mean(errors[mask])) if np.any(mask) else np.nan})
    return result


def plot_test_predictions(predictions, targets, output_dir: Path) -> list[Path]:
    """Save raw parity scatter and descriptive magnitude-bin error figures.

    Large test sets use a deterministic display subset for the scatter only.
    Every supplied row contributes to the magnitude-bin means.
    """
    prediction, target = _prediction_vectors(predictions, targets)
    shown = min(len(target), 50_000)
    indices = np.linspace(0, len(target) - 1, shown, dtype=np.int64)
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(6.5, 6.0))
        axis.scatter(target[indices], prediction[indices], s=8, alpha=.25,
                     color="#3568b0", edgecolors="none", rasterized=True)
        low = min(float(np.min(target)), float(np.min(prediction)))
        high = max(float(np.max(target)), float(np.max(prediction)))
        if high == low:
            low, high = low - .05, high + .05
        axis.plot([low, high], [low, high], color="#394553", linewidth=1.1,
                  linestyle="--", label="Perfect reconstruction")
        axis.set(title="Held-out magnitude reconstruction", xlabel="True field magnitude $|b_e|$",
                 ylabel="Raw MLP prediction")
        axis.grid(color="#e3e8ee", linewidth=.6)
        axis.legend(loc="upper left")
        caption = "Known-field simulation estimator inputs; linear predictions are not clipped."
        if shown < len(target):
            caption += f"\nScatter: {shown:,} evenly spaced rows from {len(target):,}; bin plot uses all rows."
        figure.text(.12, .025, caption, fontsize=8, color="#58677a")
        figure.subplots_adjust(bottom=.17)
        paths = _save(figure, output_dir, "test_predictions")

        bins = _binned_mae(prediction, target)
        figure, axis = plt.subplots(figsize=(7.4, 4.9))
        positions = np.arange(len(bins))
        means = [item["mae"] for item in bins]
        axis.bar(positions, means, width=.72, color="#287f75")
        labels = [f"[{item['low']:g}, {item['high']:g}{']' if i == len(bins) - 1 else ')'}"
                  for i, item in enumerate(bins)]
        axis.set_xticks(positions, labels)
        for position, item in zip(positions, bins):
            value = item["mae"] if np.isfinite(item["mae"]) else 0
            axis.annotate(f"n={item['count']:,}", (position, value), xytext=(0, 5),
                          textcoords="offset points", ha="center", fontsize=8, color="#58677a")
        axis.set(title="Reconstruction error by field magnitude", xlabel="True $|b_e|$ interval",
                 ylabel="Mean absolute error")
        axis.set_ylim(bottom=0)
        axis.margins(y=.18)
        axis.grid(axis="y", color="#e3e8ee", linewidth=.7)
        axis.set_axisbelow(True)
        caption = "Descriptive means over all test rows; edges within a Hamiltonian are correlated."
        outside = int(np.sum(target > _BIN_EDGES[-1]))
        if outside:
            caption += f"\n{outside:,} targets above 1.7 fall outside the declared paper-domain bins."
        figure.text(.12, .025, caption, fontsize=8, color="#58677a")
        figure.subplots_adjust(bottom=.19)
        paths.extend(_save(figure, output_dir, "error_by_magnitude"))
    return paths
