"""Symmetric inverse-Ising maximum pseudolikelihood for valid raw-Z snapshots.

The likelihood is for vertex pseudospins τ, with s_e=τ_u τ_v and density
proportional to exp(sum_e b_e s_e). Physical edge spins are not conditionally
independent. All edges share one jointly optimized coupling at both endpoints.
The global-flip invariant score is unchanged by reconstructing τ with τ_0=+1.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from quantum_lattice.data.snapshot_data import infer_length, reconstruct_pseudospins
from quantum_lattice.lattice import square_lattice


def _number(value, name, minimum=0.0, strict=False):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result) or result < minimum or (strict and result == minimum):
        raise ValueError(f"{name} is out of range")
    return result


def _data(states, weights):
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("inverse Ising requires nonempty (shots, edges) states")
    length = infer_length(states.shape[1])
    reconstruct_pseudospins(states, length)
    if weights is None:
        weights = np.full(len(states), 1.0 / len(states))
    else:
        if np.iscomplexobj(weights):
            raise ValueError("weights must be finite nonnegative real values")
        weights = np.asarray(weights, dtype=np.float64)
        total = float(weights.sum())
        if weights.shape != (len(states),) or not np.isfinite(weights).all() or np.any(weights < 0) or not math.isfinite(total) or total <= 0:
            raise ValueError("weights must be finite nonnegative values with positive sum")
        weights = weights / total
    incidence = square_lattice(length).vertex_edge_indices
    local = states[:, incidence].astype(np.float64)
    return states, weights, length, incidence, local


def _evaluate(fields, local, incidence, weights, ridge, hessian=True):
    # margin/2 = τ_u sum_v b_uv τ_v = sum_{e incident u} b_e s_e.
    margins = 2.0 * np.einsum("svi,vi->sv", local, fields[incidence])
    loss = float(np.sum(weights[:, None] * np.logaddexp(0.0, -margins)) / incidence.shape[0] + 0.5 * ridge * np.dot(fields, fields))
    reject = np.exp(-np.logaddexp(0.0, margins))
    gradient = ridge * fields.copy()
    matrix = ridge * np.eye(len(fields)) if hessian else None
    for vertex, edges in enumerate(incidence):
        x = local[:, vertex]
        gradient[edges] += -2.0 * (x.T @ (weights * reject[:, vertex])) / len(incidence)
        if hessian:
            factors = 4.0 * weights * reject[:, vertex] * (1.0 - reject[:, vertex]) / len(incidence)
            matrix[np.ix_(edges, edges)] += x.T @ (factors[:, None] * x)
    return loss, gradient, matrix


def pseudolikelihood_objective(states, fields, *, weights=None, ridge=1e-3):
    """Return loss, analytic gradient and Hessian for an independent oracle.

    Loss = mean_{shot,vertex} log(1+exp(-2 τ_u Σ_v b_uv τ_v))
    + ridge/2 * ||b||². Optional shot weights are normalized to sum one.
    Every input is checked for the trivial cycle sector before evaluation.
    """
    states, weights, _, incidence, local = _data(states, weights)
    ridge = _number(ridge, "ridge")
    if np.iscomplexobj(fields):
        raise ValueError("fields must be finite real edge couplings")
    fields = np.asarray(fields, dtype=np.float64)
    if fields.shape != (states.shape[1],) or not np.isfinite(fields).all():
        raise ValueError("fields must be finite real edge couplings")
    return _evaluate(fields, local, incidence, weights, ridge)


def fit_inverse_ising(states, *, weights=None, ridge=1e-3, max_iterations=100,
                      tolerance=1e-6, initial=None, callback=None):
    """Fit signed edge couplings with damped Newton/Armijo backtracking.

    ``initial`` accepts a previous ``optimizer_state`` (JSON-compatible) or a
    result containing it. ``max_iterations`` is the total ceiling, including
    previous iterations. ``callback(state)`` receives a durable-checkpoint-ready
    copy after each completed iteration; returning False pauses cleanly.
    A checkpoint is bound to exact input records, normalized weights and ridge.
    No true fields or external labels are accepted by this fitting API.
    """
    states, weights, length, incidence, local = _data(states, weights)
    ridge = _number(ridge, "ridge")
    tolerance = _number(tolerance, "tolerance", strict=True)
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    fingerprint = hashlib.sha256(states.tobytes() + weights.astype("<f8").tobytes() + str(states.shape).encode()).hexdigest()
    fields = np.zeros(states.shape[1], dtype=np.float64)
    iteration, damping = 0, 1e-8
    if initial is not None:
        initial = initial.get("optimizer_state", initial) if isinstance(initial, dict) else initial
        if not isinstance(initial, dict) or initial.get("schema_version") != 1 or initial.get("data_sha256") != fingerprint or initial.get("L") != length or initial.get("ridge") != ridge:
            raise ValueError("inverse-Ising checkpoint differs from data/topology/ridge")
        fields = np.asarray(initial.get("fields"), dtype=np.float64)
        if fields.shape != (states.shape[1],) or not np.isfinite(fields).all():
            raise ValueError("invalid inverse-Ising checkpoint fields")
        iteration = initial.get("iteration")
        if type(iteration) is not int or iteration < 0:
            raise ValueError("invalid inverse-Ising checkpoint iteration")
        damping = _number(initial.get("damping"), "checkpoint damping", strict=True)
        fields = fields.copy()

    def state(reason, loss, grad, step=0.0):
        norm = float(np.max(np.abs(grad)))
        return {"schema_version": 1, "data_sha256": fingerprint, "L": length, "ridge": ridge,
                "tolerance": tolerance, "iteration": iteration, "fields": fields.tolist(), "damping": damping,
                "objective": loss, "gradient_norm": norm, "last_step": step,
                "converged": norm <= tolerance, "stop_reason": reason}

    loss, gradient, hessian = _evaluate(fields, local, incidence, weights, ridge)
    reason = "converged" if np.max(np.abs(gradient)) <= tolerance else "max_iterations"
    checkpoint = state(reason, loss, gradient)
    while iteration < max_iterations and not checkpoint["converged"]:
        accepted = False
        for _ in range(12):
            try:
                direction = np.linalg.solve(hessian + damping * np.eye(len(fields)), gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            directional_gain = float(np.dot(gradient, direction))
            if not math.isfinite(directional_gain) or directional_gain <= 0:
                damping *= 10.0
                continue
            step = 1.0
            for _ in range(32):
                candidate = fields - step * direction
                trial, _, _ = _evaluate(candidate, local, incidence, weights, ridge, hessian=False)
                if math.isfinite(trial) and trial <= loss - 1e-4 * step * directional_gain:
                    fields = candidate
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                damping = max(1e-12, damping / 3.0)
                break
            damping *= 10.0
        if not accepted:
            checkpoint = state("line_search_failed", loss, gradient)
            break
        iteration += 1
        loss, gradient, hessian = _evaluate(fields, local, incidence, weights, ridge)
        reason = "converged" if np.max(np.abs(gradient)) <= tolerance else "max_iterations" if iteration >= max_iterations else "running"
        checkpoint = state(reason, loss, gradient, step)
        if callback is not None and callback(copy.deepcopy(checkpoint)) is False:
            checkpoint = state("paused", loss, gradient, step)
            break
    # Lists make the entire result checkpoint/log safe without custom encoders.
    return {"fields": fields.tolist(), "magnitudes": np.abs(fields).tolist(),
            "converged": checkpoint["converged"], "stop_reason": checkpoint["stop_reason"],
            "iteration": iteration, "objective": checkpoint["objective"], "gradient_norm": checkpoint["gradient_norm"],
            "optimizer_state": checkpoint}
