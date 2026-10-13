"""Run and retain the numerical calibration behind paper Figures 11 and 12."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from quantum_lattice.lattice.builders import square_lattice
from quantum_lattice.lattice.square import validate_side_length
from quantum_lattice.lattice.topology import Lattice
from quantum_lattice.metrics.calibration import (
    analytic_pattern_variance,
    evaluate_polynomial,
    exact_inverse,
    exact_vertex_rate,
    fit_inverse,
)
from quantum_lattice.simulation.phase_errors import run_calibration_point, sample_lattice_errors


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_hashes(root: Path) -> dict[str, str]:
    sources = sorted((root / "src/quantum_lattice").rglob("*.py"))
    runner = root / "experiments/phase0_figures_11_12/run.py"
    if runner.exists():
        sources.append(runner)
    return {str(path.relative_to(root)): _sha256(path) for path in sources}


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_config(config: dict) -> np.ndarray:
    """Reject unsupported protocols before starting an expensive run."""
    geometry, sampling, fit = config["geometry"], config["sampling"], config["fit"]
    if geometry["lattice"] != "square_toric" or geometry["boundary"] != "periodic":
        raise ValueError("This experiment requires a periodic square toric lattice")
    if geometry["qubits_per_cell"] != 2:
        raise ValueError("The square toric lattice has two edge qubits per cell")
    sizes = geometry["side_lengths"]
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Lattice sizes must be nonempty and distinct")
    for side in sizes:
        validate_side_length(side)
    if sampling["model"] != "independent_bernoulli_z_errors":
        raise ValueError("This calibration uses independent Bernoulli phase errors")
    if sampling["record_uncertainty_across_whole_lattice_patterns"] is not True:
        raise ValueError("Uncertainty must account for within-pattern star covariance")
    _integer(config["seed"], "seed", 0)
    _integer(sampling["independent_patterns_per_point"], "patterns", 2)
    _integer(sampling["batch_size"], "batch_size", 1)
    grid = sampling["error_probability_grid"]
    count = _integer(grid["num_points"], "num_points", 5)
    start, stop = float(grid["start"]), float(grid["stop"])
    if not np.isfinite([start, stop]).all() or not 0 <= start < stop <= 0.2:
        raise ValueError("The reproduction grid must stay in 0 <= e_r <= 0.2")
    if grid["endpoint"] is not True:
        raise ValueError("Include the declared probability-grid endpoint")
    if fit["side_length"] not in sizes:
        raise ValueError("The fitting lattice must be one of the sampled sizes")
    if fit["polynomial_degree"] != 4 or fit["zero_intercept"] is not True:
        raise ValueError("Figure 12 uses a zero-intercept fourth-degree inverse")
    if fit["method"] != "least_squares":
        raise ValueError("Only least-squares fitting is implemented")
    if fit["input"] != "mean_vertex_defect_probability" or fit["target"] != "single_qubit_error_probability":
        raise ValueError("The inverse fit maps vertex-defect probability to edge-error probability")
    if fit["compare_with_analytic_inverse"] is not True:
        raise ValueError("This reproduction includes the separately labeled analytic comparison")
    published = np.asarray(fit["published_coefficients_ascending_powers_1_to_4"], dtype=float)
    if published.shape != (4,) or not np.isfinite(published).all():
        raise ValueError("Provide four finite published coefficients")
    plotting = config["plotting"]
    if not plotting["formats"] or not set(plotting["formats"]) <= {"pdf", "svg", "png"}:
        raise ValueError("Choose pdf, svg, or png figure exports")
    _integer(plotting["png_dpi"], "png_dpi", 50)
    if plotting["x_axis"] != "vertex_defect_probability_p" or plotting["y_axis"] != "single_qubit_error_probability_e_r":
        raise ValueError("The plotted axes are p horizontally and e_r vertically")
    if plotting["analytic_check_is_separately_labeled"] is not True:
        raise ValueError("The analytical check must be labeled separately")
    return np.linspace(start, stop, count, endpoint=True)


def _fit_metrics(rows: list[dict], config: dict) -> dict:
    side = config["fit"]["side_length"]
    selected = sorted((r for r in rows if r["L"] == side), key=lambda r: r["e_r"])
    p = np.array([r["mean_p"] for r in selected])
    e = np.array([r["e_r"] for r in selected])
    result = fit_inverse(p, e)
    coefficients = result["coefficients"]
    published = config["fit"]["published_coefficients_ascending_powers_1_to_4"]
    dense_e = np.linspace(0, 0.2, 1001)
    dense_p = exact_vertex_rate(dense_e)
    result.update({
        "side_length": side,
        "published_coefficients": published,
        "sample_mse": result["diagnostics"]["mse"],
        "published_sample_mse": float(np.mean((evaluate_polynomial(p, published) - e) ** 2)),
        "exact_inverse_sample_mse": float(np.mean((exact_inverse(p) - e) ** 2)),
        "analytical_grid_mse": float(np.mean((evaluate_polynomial(dense_p, coefficients) - dense_e) ** 2)),
        "published_analytical_grid_mse": float(np.mean((evaluate_polynomial(dense_p, published) - dense_e) ** 2)),
        "max_analytical_grid_error": float(np.max(np.abs(evaluate_polynomial(dense_p, coefficients) - dense_e))),
        "description": "In-sample inverse calibration fit; not neural-network accuracy.",
        "residuals": [
            {
                "e_r": float(target), "mean_p": float(probability),
                "fresh_residual": float(evaluate_polynomial(probability, coefficients) - target),
                "published_residual": float(evaluate_polynomial(probability, published) - target),
            }
            for probability, target in zip(p, e)
        ],
        "low_rate_checks": [
            {
                "true_e_r": rate,
                "analytic_p": float(exact_vertex_rate(rate)),
                "fresh_e_r": float(evaluate_polynomial(exact_vertex_rate(rate), coefficients)),
                "published_e_r": float(evaluate_polynomial(exact_vertex_rate(rate), published)),
            }
            for rate in [0.0001, 0.001, 0.01, 0.1, 0.2]
        ],
    })
    return result


def _verification(rows: list[dict], config: dict, lattices: dict[int, Lattice]) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence(config["seed"], spawn_key=(999, 0)))
    parity_checks = {}
    for side in config["geometry"]["side_lengths"]:
        lattice = lattices[side]
        patterns = sample_lattice_errors(rng, lattice, 0.17, 256)
        parity_checks[str(side)] = bool(np.all(lattice.star_syndrome(patterns).sum(axis=-1) % 2 == 0))
    z_scores = [abs(r["z_score"]) for r in rows if r["se_p"] > 0 and r["z_score"] is not None]
    ratios = [
        r["se_p"] / float(np.sqrt(analytic_pattern_variance(r["e_r"], r["L"]) / r["patterns"]))
        for r in rows if r["se_p"] > 0
    ]
    return {
        "even_total_syndrome_parity": parity_checks,
        "zero_error_clean": all(r["mean_p"] == 0 and r["se_p"] == 0 for r in rows if r["e_r"] == 0),
        "max_absolute_standardized_deviation": max(z_scores, default=0),
        "points_outside_three_standard_errors": sum(z > 3 for z in z_scores),
        "stochastic_points": len(z_scores),
        "all_sample_means_within_five_standard_errors": all(z <= 5 for z in z_scores),
        "empirical_to_analytic_se_ratio_min": min(ratios, default=1),
        "empirical_to_analytic_se_ratio_max": max(ratios, default=1),
        "uncertainty_method": "Sample SD of independent whole-lattice fractions divided by sqrt(patterns).",
        "multiple_comparison_note": "Individual 2- or 3-SE exceedances can occur across many calibration points.",
    }


def render_run(run_dir: Path, project_root: Path) -> dict:
    """Regenerate exports from retained data without resampling."""
    from quantum_lattice.plotting.calibration import render_calibration

    root = Path(project_root).resolve()
    run_dir = (root / run_dir).resolve()
    try:
        run_dir.relative_to(root / "codex material")
    except ValueError as exc:
        raise ValueError("Generated artifacts must stay under codex material/") from exc
    config = json.loads((run_dir / "effective_config.json").read_text())
    validate_config(config)
    rows = json.loads((run_dir / "data/calibration.json").read_text())
    fit = json.loads((run_dir / "fit_metrics.json").read_text())
    files = render_calibration(
        rows, fit, run_dir / "figures",
        formats=tuple(config["plotting"]["formats"]), dpi=config["plotting"]["png_dpi"],
    )
    presentation = run_dir / "presentation"
    presentation.mkdir(exist_ok=True)
    for category in ("presentation", "presentation_previews"):
        moved = []
        for name in files[category]:
            source = Path(name)
            destination = presentation / source.name
            source.replace(destination)
            moved.append(str(destination))
        files[category] = moved
    manifest = {
        category: [str(Path(name).relative_to(run_dir)) for name in names]
        for category, names in files.items()
    }
    _write_json(run_dir / "artifact_manifest.json", manifest)
    _write_json(run_dir / "render_provenance.json", {
        "rendered_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git(root, "status", "--porcelain")),
        "source_sha256": _source_hashes(root),
        "input_sha256": {
            name: _sha256(run_dir / name)
            for name in ("effective_config.json", "data/calibration.json", "fit_metrics.json")
        },
        "note": "Exports regenerated from retained data; original sampling provenance is unchanged.",
    })
    _write_run_readme(run_dir, config)
    return manifest


def run_experiment(
    config: dict, project_root: Path, run_id: str | None = None, progress=None, render: bool = True
) -> Path:
    """Sample, fit, check, and retain a complete independently seeded run."""
    error_rates = validate_config(config)
    root = Path(project_root).resolve()
    output_root = (root / config["output_root"]).resolve()
    try:
        output_root.relative_to(root / "codex material")
    except ValueError as exc:
        raise ValueError("Generated artifacts must stay under codex material/") from exc
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-seed{config['seed']}"
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one directory name")
    # Reuse one immutable topology per size across every probability and batch.
    lattices = {side: square_lattice(side) for side in config["geometry"]["side_lengths"]}
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "data").mkdir()
    (run_dir / "INCOMPLETE").write_text("Run has not completed.\n")
    _write_json(run_dir / "effective_config.json", config)
    _write_json(run_dir / "lattices.json", {
        str(side): {
            "geometry": lattice.geometry, "boundary": lattice.boundary,
            "shape": list(lattice.shape),
            "n_vertices": lattice.n_vertices, "n_edges": lattice.n_edges,
            "n_faces": lattice.n_faces,
            "vertex_degrees": sorted(set(map(int, lattice.vertex_degrees))),
            "face_degrees": sorted(set(map(int, lattice.face_degrees))),
            "edge_id_order": "2*(y*L+x)+orientation; horizontal=0, vertical=1",
            "sample_shape": "(batch, n_edges)",
        }
        for side, lattice in lattices.items()
    })
    import matplotlib
    _write_json(run_dir / "environment.json", {
        "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "bit_generator": "PCG64",
    })
    paper = root / "Reference Paper.pdf"
    _write_json(run_dir / "provenance.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git(root, "status", "--porcelain")),
        "source_sha256": _source_hashes(root),
        "paper_sha256": _sha256(paper) if paper.exists() else None,
        "seed": config["seed"],
        "stream_scheme": "SeedSequence(seed, spawn_key=(L, probability_grid_index))",
        "original_grid_and_counts_available": False,
        "method": "Fresh independent Bernoulli error patterns; no Metropolis ground-state sampling.",
        "lattice_representation": "Immutable Vertex/Edge/Face/Lattice topology with compiled incidence arrays.",
    })
    start = time.perf_counter()
    rows = []
    settings = config["sampling"]
    for side in config["geometry"]["side_lengths"]:
        for index, rate in enumerate(error_rates):
            sequence = np.random.SeedSequence(config["seed"], spawn_key=(side, index))
            row = run_calibration_point(
                side, float(rate), settings["independent_patterns_per_point"],
                settings["batch_size"], sequence,
                lattice=lattices[side],
            )
            row["probability_grid_index"] = index
            row["analytic_se_p"] = float(np.sqrt(analytic_pattern_variance(rate, side) / row["patterns"]))
            rows.append(row)
        if progress:
            progress(f"L={side}: {len(error_rates)} calibration points sampled")
    sampling_seconds = time.perf_counter() - start
    _write_json(run_dir / "data/calibration.json", rows)
    fields = ["L", "e_r", "patterns", "mean_p", "se_p", "analytic_p", "analytic_se_p",
              "z_score", "mean_stabilizer", "probability_grid_index", "stream_spawn_key"]
    with (run_dir / "data/calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "stream_spawn_key": json.dumps(row["stream_spawn_key"])})
    _write_json(run_dir / "fit_metrics.json", _fit_metrics(rows, config))
    verification = _verification(rows, config, lattices)
    _write_json(run_dir / "verification.json", verification)
    if not all(verification["even_total_syndrome_parity"].values()) or not verification["zero_error_clean"]:
        raise RuntimeError("A deterministic geometry check failed")
    if not verification["all_sample_means_within_five_standard_errors"]:
        raise RuntimeError("Inspect calibration data: a sample mean is beyond 5 standard errors")
    if render:
        render_run(run_dir, root)
    _write_json(run_dir / "timing.json", {
        "sampling_seconds": sampling_seconds,
        "total_seconds": time.perf_counter() - start,
        "numerical_points": len(rows),
        "independent_error_patterns": sum(r["patterns"] for r in rows),
    })
    _write_run_readme(run_dir, config)
    (run_dir / "INCOMPLETE").unlink()
    return run_dir


def _write_run_readme(run_dir: Path, config: dict) -> None:
    fit = json.loads((run_dir / "fit_metrics.json").read_text())
    check = json.loads((run_dir / "verification.json").read_text())
    grid = config["sampling"]["error_probability_grid"]
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    presentation_links = []
    for category in ("presentation", "figure_11", "figure_12"):
        for relative in manifest.get(category, []):
            if relative.endswith(".pdf"):
                label = "Two-page professor presentation" if category == "presentation" else category.replace("_", " ").title()
                presentation_links.append(f"[{label}]({relative})")
    export_note = "\n\n".join(presentation_links) or "See artifact_manifest.json for generated exports."
    if not manifest:
        export_note = "This data-only run has not generated figures. Use --render-only to export retained data."
    material_root = next(parent for parent in run_dir.parents if parent.name == "codex material")
    relative_run = run_dir.relative_to(material_root.parent)
    text = f"""# Figures 11–12: fresh numerical reproduction

Source: Valenti et al., *Hamiltonian learning for quantum error correction*,
Appendix D.1, DOI 10.1103/PhysRevResearch.1.033092.

Each edge receives a Z error independently with probability e_r. A vertex is
defective when its four incident errors have odd parity. This is independent
error-pattern calibration, not a ground-state Metropolis simulation.

- Lattice sizes: {config['geometry']['side_lengths']}.
- Grid: {grid['num_points']} probabilities from {grid['start']} to {grid['stop']}.
- Independent lattice patterns per point: {config['sampling']['independent_patterns_per_point']:,}.
- Base seed: {config['seed']}; stream keys are retained with every data row.
- Fresh inverse-fit MSE: {fit['sample_mse']:.8g}.
- Published-polynomial MSE on our L={fit['side_length']} data: {fit['published_sample_mse']:.8g}.
- Maximum sample deviation from the exact forward relation: {check['max_absolute_standardized_deviation']:.3f} standard errors.

## Presentation

{export_note}

The artifact manifest lists all generated formats and page previews.
All axes use p horizontally and e_r vertically; the original Figure 11 caption
reverses them. Numerical samples, the fresh fit, published D4, and the derived
analytical check are labeled separately.

## Reproduction scope

The original calibration data, grid, and trial count are unavailable. These are
our declared settings, not recovered historical samples. The fit MSE concerns
this calibration, not neural-network accuracy. The converted rate is an
equivalent independent-error summary for generally correlated states.

Numeric data are in data/calibration.csv and data/calibration.json.
Fit coefficients/residuals, uncertainty checks, config, environment, provenance,
and timing are retained beside this note.

## Rerun

From the repository root with the project environment active, create a new run
using the retained configuration (the default timestamp prevents overwriting):

```sh
python experiments/phase0_figures_11_12/run.py --config "{relative_run}/effective_config.json"
```

Regenerate exports without resampling:

```sh
python experiments/phase0_figures_11_12/run.py --render-only "{relative_run}"
```

provenance.json describes the original sampling source. render_provenance.json
records the current export source and hashes of the retained inputs separately.
"""
    (run_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase0_figures_11_12.json")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id")
    parser.add_argument("--render-only", type=Path, help="Regenerate plots from an existing run")
    arguments = parser.parse_args(argv)
    root = Path(arguments.project_root).resolve()
    if arguments.render_only is not None:
        render_run(arguments.render_only, root)
        print(f"Rendered retained data in {arguments.render_only}", flush=True)
        return
    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = json.loads(config_path.read_text())
    directory = run_experiment(
        config, root, arguments.run_id, progress=lambda message: print(message, flush=True)
    )
    print(f"Completed: {directory}", flush=True)


if __name__ == "__main__":
    main()
