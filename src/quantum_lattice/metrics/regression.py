"""Raw magnitude-regression metrics with Hamiltonian-grouped uncertainty."""

from __future__ import annotations

import numpy as np


def _vectors(predictions, y, hamiltonian_ids, edge_ids):
    prediction, target = np.asarray(predictions, dtype=float), np.asarray(y, dtype=float)
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    names, edges = np.asarray(hamiltonian_ids), np.asarray(edge_ids)
    if prediction.ndim != 1 or target.shape != prediction.shape or not len(target):
        raise ValueError("predictions and targets must be nonempty matching vectors or (n,1) arrays")
    if names.shape != target.shape or edges.shape != target.shape:
        raise ValueError("Hamiltonian/edge IDs must align with prediction rows")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("predictions must be finite and magnitude targets finite and nonnegative")
    if edges.dtype.kind not in "iu" or np.any(edges < 0) or names.dtype.kind not in "US":
        raise ValueError("edge IDs must be nonnegative integers and Hamiltonian IDs strings")
    return prediction, target, names.astype(str), edges


def _metrics(errors):
    mse = float(np.mean(np.square(errors)))
    return {"count": len(errors), "mse": mse, "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(errors)))}


def per_hamiltonian_metrics(predictions, y, hamiltonian_ids, edge_ids):
    prediction, target, names, _ = _vectors(predictions, y, hamiltonian_ids, edge_ids)
    errors = prediction - target
    ids, inverse = np.unique(names, return_inverse=True)
    counts = np.bincount(inverse)
    absolute = np.bincount(inverse, weights=np.abs(errors))
    squared = np.bincount(inverse, weights=errors * errors)
    return [{"hamiltonian_id": str(name), "count": int(count),
             "mse": float(sq / count), "rmse": float(np.sqrt(sq / count)),
             "mae": float(ab / count)}
            for name, count, sq, ab in zip(ids, counts, squared, absolute)]


def _bootstrap(errors, names, samples, seed):
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    ids, inverse = np.unique(names, return_inverse=True)
    if samples == 0:
        return {"method": "whole-Hamiltonian bootstrap", "samples": 0,
                "hamiltonians": len(ids), "confidence_level": 0.95, "intervals": None}
    count = np.bincount(inverse)
    absolute = np.bincount(inverse, weights=np.abs(errors))
    squared = np.bincount(inverse, weights=errors * errors)
    random = np.random.default_rng(seed)
    mae, mse = np.empty(samples), np.empty(samples)
    # Sample identities, retaining every row of each selected Hamiltonian.
    for i in range(samples):
        selected = random.integers(0, len(ids), size=len(ids))
        n = np.sum(count[selected])
        mae[i] = np.sum(absolute[selected]) / n
        mse[i] = np.sum(squared[selected]) / n
    intervals = {key: np.quantile(value, [0.025, 0.975]).tolist()
                 for key, value in (("mse", mse), ("rmse", np.sqrt(mse)), ("mae", mae))}
    return {"method": "whole-Hamiltonian bootstrap", "samples": samples,
            "hamiltonians": len(ids), "confidence_level": 0.95, "intervals": intervals,
            "caveat": "Groups are resampled, not edges; this excludes model-training and Monte Carlo bias."}


def regression_report(predictions, y, hamiltonian_ids, edge_ids, fixed_edge=0,
                      training_target_mean=None, bootstrap_samples=200, seed=0):
    """Evaluate raw predictions; never clip predictions or estimated inputs.

    Confidence intervals account for correlated edge errors by resampling whole
    Hamiltonians. The constant baseline must use the training target mean.
    """
    prediction, target, names, edges = _vectors(predictions, y, hamiltonian_ids, edge_ids)
    if isinstance(fixed_edge, bool) or not isinstance(fixed_edge, (int, np.integer)) or fixed_edge < 0:
        raise ValueError("fixed_edge must be a nonnegative integer")
    errors = prediction - target
    report = {**_metrics(errors), "hamiltonians": len(np.unique(names)),
              "prediction_policy": "raw, unclipped linear outputs",
              "negative_prediction_fraction": float(np.mean(prediction < 0)),
              "above_1_7_prediction_fraction": float(np.mean(prediction > 1.7)),
              "per_hamiltonian": per_hamiltonian_metrics(prediction, target, names, edges),
              "bootstrap": _bootstrap(errors, names, bootstrap_samples, seed)}
    mask = edges == fixed_edge
    report["fixed_edge"] = {"edge_id": int(fixed_edge),
                            "metrics": _metrics(errors[mask]) if np.any(mask) else None}
    boundaries = [0.0, 0.1, 0.3, 0.6, 1.0, 1.4, 1.7]
    bins = []
    for i, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (target >= low) & (target <= high if i == len(boundaries) - 2 else target < high)
        bins.append({"low": low, "high": high, "upper_inclusive": i == len(boundaries) - 2,
                     "metrics": _metrics(errors[mask]) if np.any(mask) else None})
    report["target_magnitude_bins"] = bins
    report["targets_outside_bins"] = int(np.sum((target < boundaries[0]) | (target > boundaries[-1])))
    if training_target_mean is not None:
        if not np.isfinite(training_target_mean) or training_target_mean < 0:
            raise ValueError("training_target_mean must be a finite nonnegative scalar")
        report["constant_training_mean_baseline"] = {
            "prediction": float(training_target_mean), **_metrics(float(training_target_mean) - target)}
    return report
