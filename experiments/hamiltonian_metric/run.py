#!/usr/bin/env python3
"""Evaluate the paper's D5 on signed simulated field vectors, or run a demo."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from quantum_lattice.lattice import square_lattice
from quantum_lattice.metrics.hamiltonian import pauli_coefficients, normalized_pauli_distance

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "codex material/artifacts/hamiltonian_metric"


def evaluate(data):
    """One Hamiltonian pair; no model selection, sign inference or file access."""
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("input must declare schema_version=1")
    expected = {"schema_version", "L", "model", "true_b_z", "recovered_b_z",
                "true_b_x", "recovered_b_x", "sign_source", "description"}
    if set(data) - expected:
        raise ValueError("input contains unknown keys")
    lattice = square_lattice(data.get("L"))
    model = data.get("model", "two_channel")
    sign_source = data.get("sign_source")
    if sign_source not in {"provided_signed_fields", "reference_sign_oracle", "synthetic_signed_fields"}:
        raise ValueError("declare sign_source: provided_signed_fields, reference_sign_oracle, or synthetic_signed_fields")
    for key in ("true_b_z", "recovered_b_z"):
        if key not in data:
            raise ValueError(f"{key} is required")
    reference = pauli_coefficients(lattice, data["true_b_z"], data.get("true_b_x"), model=model)
    estimate = pauli_coefficients(lattice, data["recovered_b_z"], data.get("recovered_b_x"), model=model)
    centered_reference = pauli_coefficients(lattice, data["true_b_z"], data.get("true_b_x"), model=model, include_identity=False)
    centered_estimate = pauli_coefficients(lattice, data["recovered_b_z"], data.get("recovered_b_x"), model=model, include_identity=False)
    errors = {}
    for axis in ("z", "x"):
        true = np.asarray(data.get(f"true_b_{axis}", np.zeros(lattice.n_edges)), dtype=np.float64)
        recovered = np.asarray(data.get(f"recovered_b_{axis}", np.zeros(lattice.n_edges)), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            difference = true - recovered
        if not np.isfinite(difference).all():
            raise ValueError("field difference exceeds finite float64 range")
        scale = float(np.max(np.abs(difference)))
        errors[axis] = {"rmse": scale * float(np.sqrt(np.mean((difference/scale)**2))) if scale else 0.0,
                        "mae": scale * float(np.mean(np.abs(difference/scale))) if scale else 0.0}
    return {
        "schema_version": 1, "scope": "signed known-field simulation; one Hamiltonian reconstruction",
        "reference": "Valenti et al., Phys. Rev. Research 1, 033092 (2019), Eq9/D5-D9",
        "L": data["L"], "qubits": lattice.n_edges, "model": model, "sign_source": sign_source,
        "basis": "distinct global Pauli strings; all equal strings added before normalization",
        "fixed_stabilizers": "included", "distance": "unsquared Euclidean; mathematical range [0,2]",
        "figure_rescaling": "none; historical plot convention is not recovered",
        "d5_full_hamiltonian": normalized_pauli_distance(reference, estimate),
        "d5_traceless_hamiltonian": normalized_pauli_distance(centered_reference, centered_estimate),
        "traceless_definition": "same metric after removing the single aggregated identity coefficient",
        "field_errors": errors,
        "coefficient_counts": {"true": len(reference.coefficients), "recovered": len(estimate.coefficients)},
        "log_scales": {"true": reference.log_scale, "recovered": estimate.log_scale},
        "prediction_policy": "signed values supplied unchanged; no abs, clipping, or inferred signs",
        "recovered_field_role": "estimated original fields, not the remaining residual after subtraction",
        "physical_measurement_protocol_required": False,
    }


def demo():
    rng = np.random.default_rng(20261018)
    true = rng.uniform(-.8, .8, 18)
    return {"schema_version": 1, "L": 3, "model": "two_channel",
            "true_b_z": true.tolist(), "recovered_b_z": (.8*true).tolist(),
            "sign_source": "synthetic_signed_fields",
            "description": "Deterministic demonstration only: estimated Z fields are 80% of true signed fields; X fields zero. Not an MLP result."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON with signed true and recovered fields; omitted runs a deterministic demo")
    parser.add_argument("--output", type=Path, help="new directory below codex material/artifacts/hamiltonian_metric/")
    args = parser.parse_args(argv)
    data = json.loads(args.input.read_text()) if args.input else demo()
    result = evaluate(data)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ARTIFACT_ROOT / f"{'evaluation' if args.input else 'demo'}-{stamp}").resolve()
    if output == ARTIFACT_ROOT.resolve() or not output.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("output must be a new subdirectory of the ignored Hamiltonian-metric artifact root")
    if output.exists():
        raise ValueError("output already exists; choose a new directory to preserve earlier results")
    output.mkdir(parents=True)
    (output/"input.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    (output/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    provenance = {"input_path": str(args.input.resolve()) if args.input else None,
                  "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest() if args.input else None,
                  "demo": args.input is None, "source_hashes": {
                      str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (Path(__file__), ROOT/"src/quantum_lattice/metrics/hamiltonian.py")},
                  "numpy": np.__version__, "python": sys.version,
                  "training_or_monte_carlo_performed": False}
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    report = ("# Hamiltonian coefficient-distance evaluation\n\n"
              + ("**Illustrative deterministic example; not a trained-model result.**\n\n" if args.input is None else "")
              + f"L={result['L']}; {result['qubits']} qubits; model `{result['model']}`; signs `{result['sign_source']}`.\n\n"
              + f"- D5 with identity and fixed stabilizers: **{result['d5_full_hamiltonian']:.10g}**.\n"
              + f"- Traceless D5, fixed stabilizers retained: **{result['d5_traceless_hamiltonian']:.10g}**.\n"
              + f"- Z-field RMSE: **{result['field_errors']['z']['rmse']:.10g}**.\n\n"
              + "Equal global Pauli strings are merged. Values are unsquared, with no figure-specific normalization. "
              "Recovered fields mean estimated original fields, not residual fields. "
              "The two identity conventions must not be mixed in comparisons. "
              "This implements the stated D5 formula with an explicit global basis; "
              "historical figure-generation conventions are not fully available.\n\n"
              + "Files: `input.json` retains exact signed vectors/settings; `metrics.json` the scores/conventions; "
              "`provenance.json` input/source hashes and environment. No training, sampling or sign inference occurred.\n")
    (output/"RESULTS.md").write_text(report)
    print("Deterministic demonstration (not a network result)." if args.input is None else "Signed-field reconstruction comparison.")
    print(f"D5 full Hamiltonian: {result['d5_full_hamiltonian']:.10g}")
    print(f"D5 traceless Hamiltonian: {result['d5_traceless_hamiltonian']:.10g}")
    print(f"Results: {output/'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"Hamiltonian metric stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
