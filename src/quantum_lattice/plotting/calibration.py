"""Publication and presentation figures for independent-error calibration.

All curve coordinates come from numerical whole-lattice error patterns. The
closed-form relation is an explicitly labeled check, not replacement data.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import textwrap
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from quantum_lattice.metrics.calibration import (
    exact_inverse as _inverse,
    evaluate_polynomial as _polynomial,
    exact_vertex_rate,
)


_PALETTE = (
    "#3568B0", "#27A3AE", "#D59A28", "#D66D3B", "#8556A3", "#489274", "#BA536A"
)
_MARKERS = ("o", "s", "^", "D", "v", "P", "X")
_INK = "#243247"
_MUTED = "#58677A"
_FIT = "#285EAD"
_PUBLISHED = "#B3652E"
_DATA = "#287F75"
_EXACT = "#394553"
_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "axes.labelcolor": _INK,
    "axes.edgecolor": "#9CA7B4",
    "axes.linewidth": 0.7,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "dejavusans",
}


def _group_rows(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, np.ndarray]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["L"])].append(row)
    if not grouped:
        raise ValueError("At least one numerical calibration row is required")
    arrays = {}
    for side, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: float(row["e_r"]))
        arrays[side] = {
            key: np.asarray([row[key] for row in ordered], dtype=float)
            for key in ("e_r", "mean_p", "se_p", "patterns", "analytic_p", "z_score")
        }
        if np.any(np.diff(arrays[side]["e_r"]) <= 0):
            raise ValueError(f"Error probabilities must be distinct for L={side}")
        if np.any(arrays[side]["se_p"] < 0):
            raise ValueError("Standard errors must be nonnegative")
    return arrays


def _clean_axis(axis: Any, *, percent_y: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E3E8EE", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    if percent_y:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def _draw_size_curves(axis: Any, groups: dict[int, dict[str, np.ndarray]]) -> None:
    for index, (side, data) in enumerate(groups.items()):
        # Stagger marker positions so overlapping curves remain identifiable.
        axis.plot(
            data["mean_p"], data["e_r"],
            color=_PALETTE[index % len(_PALETTE)], linewidth=1.35,
            marker=_MARKERS[index % len(_MARKERS)], markersize=4.1,
            markevery=(index % 7, 7), label=rf"$L={side}$",
        )
    largest = groups[max(groups)]
    error_grid = np.linspace(0.0, max(float(d["e_r"].max()) for d in groups.values()), 500)
    analytic_p = exact_vertex_rate(error_grid)
    axis.plot(
        analytic_p, error_grid, color=_EXACT, linewidth=1.2,
        linestyle=(0, (5, 3)), label="Analytical check", zorder=2.5,
    )
    axis.set_xlim(0, max(float(d["mean_p"].max()) for d in groups.values()) * 1.045)
    axis.set_ylim(0, float(largest["e_r"].max()) * 1.08)
    axis.set_xlabel(r"Vertex-defect probability $p$")
    axis.set_ylabel(r"Single-qubit phase-error probability $e_r$")
    _clean_axis(axis)
    axis.legend(loc="upper left", ncol=2, columnspacing=1.25, handlelength=2.4)


def _draw_size_residuals(axis: Any, groups: dict[int, dict[str, np.ndarray]], *, compact: bool = False) -> None:
    axis.axhspan(-2, 2, color="#EAF0F6", alpha=0.9, zorder=0)
    axis.axhline(0, color="#8794A3", linewidth=0.8, zorder=1)
    maximum = 0.0
    for index, (_, data) in enumerate(groups.items()):
        valid = data["se_p"] > 0
        z_score = (data["mean_p"][valid] - data["analytic_p"][valid]) / data["se_p"][valid]
        maximum = max(maximum, float(np.max(np.abs(z_score))) if z_score.size else 0.0)
        axis.plot(
            data["analytic_p"][valid], z_score,
            color=_PALETTE[index % len(_PALETTE)],
            marker=_MARKERS[index % len(_MARKERS)], markersize=2.9 if compact else 3.4,
            linewidth=0.55, alpha=0.85,
        )
    bound = max(3.2, maximum * 1.12)
    axis.set_ylim(-bound, bound)
    axis.set_xlim(0, max(float(d["analytic_p"].max()) for d in groups.values()) * 1.045)
    axis.set_xlabel(r"Analytical vertex-defect probability $p$")
    axis.set_ylabel("Deviation / SE" if compact else "Numerical − analytical\n(in standard errors)")
    _clean_axis(axis)
    axis.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    if compact:
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)


def _draw_fit(
    axis: Any,
    data: dict[str, np.ndarray],
    coefficients: np.ndarray,
    published: np.ndarray,
) -> None:
    p_grid = np.linspace(0.0, float(data["mean_p"].max()), 600)
    axis.plot(p_grid, _polynomial(p_grid, coefficients), color=_FIT, linewidth=2.0, label="Fresh fourth-degree fit")
    axis.plot(p_grid, _polynomial(p_grid, published), color=_PUBLISHED, linewidth=1.6,
              linestyle=(0, (6, 3)), label="Published polynomial (D4)")
    axis.plot(p_grid, _inverse(p_grid), color=_EXACT, linewidth=1.4,
              linestyle=(0, (1, 2)), label="Exact independent-error inverse")
    axis.errorbar(
        data["mean_p"], data["e_r"], xerr=1.96 * data["se_p"],
        fmt="o", color=_DATA, markersize=3.6, markeredgecolor="white",
        markeredgewidth=0.55, elinewidth=0.65, capsize=0,
        label="Numerical data (95% intervals)", zorder=5,
    )
    axis.set_xlim(0.0, float(data["mean_p"].max()) * 1.045)
    axis.set_ylim(0, float(data["e_r"].max()) * 1.08)
    axis.set_xlabel(r"Vertex-defect probability $p$")
    axis.set_ylabel(r"Single-qubit phase-error probability $e_r$")
    _clean_axis(axis)
    axis.legend(loc="upper left", handlelength=2.7)


def _draw_fit_residuals(
    axis: Any,
    data: dict[str, np.ndarray],
    coefficients: np.ndarray,
    published: np.ndarray,
    *, compact: bool = False,
) -> None:
    p_grid = np.linspace(0.0, float(data["mean_p"].max()), 600)
    exact = _inverse(p_grid)
    axis.axhline(0, color="#8794A3", linewidth=0.8)
    axis.plot(p_grid, 100 * (_polynomial(p_grid, coefficients) - exact), color=_FIT, linewidth=1.6)
    axis.plot(p_grid, 100 * (_polynomial(p_grid, published) - exact), color=_PUBLISHED,
              linewidth=1.3, linestyle=(0, (6, 3)))
    # Delta-method propagation is sufficient for these tiny horizontal CIs.
    derivative = 0.25 * np.maximum(1 - 2 * data["mean_p"], np.finfo(float).eps) ** -0.75
    axis.errorbar(
        data["mean_p"], 100 * (data["e_r"] - _inverse(data["mean_p"])),
        yerr=100 * 1.96 * derivative * data["se_p"],
        fmt="o", color=_DATA, markersize=2.8 if compact else 3.4,
        elinewidth=0.6, capsize=1.2, markeredgewidth=0.4,
    )
    axis.set_xlim(0, float(data["mean_p"].max()) * 1.045)
    axis.set_xlabel(r"Vertex-defect probability $p$")
    axis.set_ylabel("Residual (percentage points)" if compact else "Difference from exact inverse\n(percentage points)")
    _clean_axis(axis)
    if compact:
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)


def _standalone_size(groups: dict[int, dict[str, np.ndarray]]) -> Any:
    fig, (main, residual) = plt.subplots(
        2, 1, figsize=(9.0, 7.6), gridspec_kw={"height_ratios": [3.0, 1.15]},
    )
    fig.subplots_adjust(left=0.115, right=0.975, top=0.84, bottom=0.105, hspace=0.48)
    fig.suptitle("Figure 11 reproduction", x=0.115, y=0.965, ha="left", fontsize=20, color=_INK, weight="semibold")
    fig.text(0.115, 0.915, "Independent phase errors give the same calibration across lattice sizes", fontsize=11, color=_MUTED)
    _draw_size_curves(main, groups)
    _draw_size_residuals(residual, groups)
    residual.set_title("Numerical check; shaded band denotes ±2 standard errors", loc="left", fontsize=9.5, pad=6)
    return fig


def _standalone_fit(data: dict[str, np.ndarray], coefficients: np.ndarray, published: np.ndarray, side: int) -> Any:
    fig, (main, residual) = plt.subplots(
        2, 1, figsize=(9.0, 7.6), gridspec_kw={"height_ratios": [3.0, 1.15]},
    )
    fig.subplots_adjust(left=0.115, right=0.975, top=0.84, bottom=0.105, hspace=0.48)
    fig.suptitle("Figure 12 reproduction", x=0.115, y=0.965, ha="left", fontsize=20, color=_INK, weight="semibold")
    fig.text(0.115, 0.915, rf"Fresh numerical calibration at $L={side}$ and a new inverse polynomial", fontsize=11, color=_MUTED)
    _draw_fit(main, data, coefficients, published)
    _draw_fit_residuals(residual, data, coefficients, published)
    residual.set_title("Residuals reveal differences hidden by overlapping curves", loc="left", fontsize=9.5, pad=6)
    return fig


def _caption(fig: Any, text: str) -> None:
    fig.text(0.075, 0.184, "METHOD AND INTERPRETATION", fontsize=8.5, weight="bold", color=_MUTED)
    fig.text(0.075, 0.156, textwrap.fill(text, width=155), fontsize=9.5, color=_INK,
             va="top", linespacing=1.35)


def _presentation_page(
    index: int,
    groups: dict[int, dict[str, np.ndarray]],
    coefficients: np.ndarray,
    published: np.ndarray,
    fit_metrics: dict[str, Any],
    fit_side: int,
) -> Any:
    fig = plt.figure(figsize=(13.333333, 7.5))
    fig.text(0.075, 0.939, "TORIC-CODE ERROR CALIBRATION", fontsize=9.5, weight="bold", color=_MUTED)
    titles = {
        1: "Figure 11: Size-independent defect calibration",
        2: "Figure 12: Inferring a phase-error rate",
    }
    fig.text(0.075, 0.884, titles[index], fontsize=21, weight="semibold", color=_INK)
    fig.text(0.955, 0.939, f"{index} / 2", fontsize=9.5, ha="right", color=_MUTED)
    main = fig.add_axes((0.075, 0.285, 0.62, 0.52))
    residual = fig.add_axes((0.775, 0.475, 0.195, 0.245))
    fig.text(0.755, 0.776, "A check below the curve", fontsize=12, weight="semibold", color=_INK)
    pattern_counts = sorted({int(count) for data in groups.values() for count in data["patterns"]})
    pattern_text = f"{pattern_counts[0]:,}" if len(pattern_counts) == 1 else "variable numbers of"
    largest = groups[max(groups)]
    point_count = len(largest["e_r"])
    low = min(float(data["e_r"].min()) for data in groups.values())
    high = max(float(data["e_r"].max()) for data in groups.values())
    if index == 1:
        _draw_size_curves(main, groups)
        _draw_size_residuals(residual, groups, compact=True)
        fig.text(0.755, 0.374, "Uncertainty uses independent whole-lattice\npatterns, not independent vertices.\nThe band marks ±2 standard errors.", fontsize=9.0, color=_MUTED, linespacing=1.5, va="top")
        fig.text(0.755, 0.270, r"$p=[1-(1-2e_r)^4]/2$", fontsize=14, color=_INK)
        caption = (
            f"Each edge independently receives a Z error with probability e_r. A star is defective when an odd number of its four edges have errors. "
            f"At each of {point_count} probabilities from {low:g} to {high:g}, we sample {pattern_text} independent lattice patterns per size. "
            "Numerical size curves overlap; the dashed relation is a separate analytical check. Axes follow the original plot (p horizontally, e_r vertically), "
            "correcting its caption inconsistency. Grid, sample count, and seeds are our declared choices; the historical settings were not supplied."
        )
    else:
        data = groups[fit_side]
        _draw_fit(main, data, coefficients, published)
        _draw_fit_residuals(residual, data, coefficients, published, compact=True)
        mse = float(fit_metrics["sample_mse"])
        fig.text(0.755, 0.381, f"Fresh-fit MSE: {mse:.3g}\nMeasured against the numerical points.\nThis is not neural-network accuracy.", fontsize=9.0, color=_MUTED, linespacing=1.5, va="top")
        fig.text(0.755, 0.271, r"$p=(1-\overline{\langle A\rangle})/2$", fontsize=14, color=_INK)
        caption = (
            f"Fit a zero-intercept fourth-degree inverse polynomial to the {len(data['e_r'])} numerical points at L={fit_side}; "
            f"each point uses {pattern_text} independent error patterns. The fresh fit, published D4, and exact independent-error inverse are compared; "
            "the residual panel exposes their small differences. The original correction scripts use D4 to convert mean stabilizer defects into a reported "
            "single-qubit error rate. This is a calibration/reporting step, not the network correction rule. For correlated states, it is an equivalent independent-error rate."
        )
    _caption(fig, caption)
    fig.text(0.075, 0.030, "Reference: Valenti et al., Appendix D.1 · Fresh numerical reproduction", fontsize=8.5, color=_MUTED)
    return fig


def render_calibration(
    rows: list[dict[str, Any]],
    fit_metrics: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
    dpi: int = 300,
) -> dict[str, list[str]]:
    """Export numerical Figures 11–12 and a two-page landscape presentation.

    ``rows`` must include ``L``, ``e_r``, ``patterns``, ``mean_p``, ``se_p``,
    ``analytic_p`` and ``z_score``. ``fit_metrics`` supplies ``coefficients``
    and ``published_coefficients`` in ascending powers one through four, and
    ``sample_mse``. Optional ``side_length`` selects the fitting lattice;
    otherwise the largest supplied size is used. Returned paths are JSON-safe.
    """
    groups = _group_rows(rows)
    coefficients = np.asarray(fit_metrics["coefficients"], dtype=float)
    published = np.asarray(fit_metrics["published_coefficients"], dtype=float)
    if coefficients.shape != (4,) or published.shape != (4,):
        raise ValueError("Calibration fits require four ascending coefficients")
    fit_side = int(fit_metrics.get("side_length", max(groups)))
    if fit_side not in groups:
        raise ValueError(f"No numerical data for fitting lattice L={fit_side}")
    allowed = {"pdf", "svg", "png"}
    if not set(formats) <= allowed:
        raise ValueError(f"Supported figure formats: {sorted(allowed)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {"figure_11": [], "figure_12": [], "presentation": [], "presentation_previews": []}
    with plt.rc_context(_STYLE):
        figures = {
            "figure_11": _standalone_size(groups),
            "figure_12": _standalone_fit(groups[fit_side], coefficients, published, fit_side),
        }
        for name, fig in figures.items():
            for extension in formats:
                path = output_dir / f"{name}.{extension}"
                fig.savefig(path, dpi=dpi)
                manifest[name].append(str(path))
            plt.close(fig)
        pdf_path = output_dir / "figures_11_12_presentation.pdf"
        with PdfPages(pdf_path, metadata={
            "Title": "Toric-code independent-error calibration: Figures 11 and 12",
            "Subject": "Fresh numerical reproduction and inverse-polynomial comparison",
        }) as pdf:
            for page_index in (1, 2):
                fig = _presentation_page(page_index, groups, coefficients, published, fit_metrics, fit_side)
                pdf.savefig(fig)
                preview = output_dir / f"presentation_page_{page_index}.png"
                fig.savefig(preview, dpi=150)
                manifest["presentation_previews"].append(str(preview))
                plt.close(fig)
        manifest["presentation"].append(str(pdf_path))
    return manifest
