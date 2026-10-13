"""Known-field amplitude-ratio observables and tiny-orbit exact checks.

These estimators use the field values of the solvable simulation. In
particular, star estimators are not ±1 outcomes of X-basis measurements.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from quantum_lattice.lattice import Lattice
from quantum_lattice.simulation.solvable import _fields, _nonnegative_integer


@lru_cache(maxsize=16)
def _shared_incidence(lattice: Lattice) -> tuple[np.ndarray, np.ndarray]:
    # With parallel edges, two stars share every edge between their endpoints.
    groups: dict[tuple[int, int], list[int]] = {}
    for edge in lattice.edges:
        groups.setdefault(tuple(sorted(edge.vertex_ids)), []).append(edge.id)
    width = max((len(group) for group in groups.values()), default=0)
    indices = np.zeros((lattice.n_edges, width), dtype=np.int64)
    mask = np.zeros(indices.shape, dtype=bool)
    for edge in lattice.edges:
        group = groups[tuple(sorted(edge.vertex_ids))]
        indices[edge.id, : len(group)] = group
        mask[edge.id, : len(group)] = True
    return indices, mask


def local_estimators(lattice: Lattice, fields: Any, spins: Any) -> dict[str, np.ndarray]:
    """Return all stars, neighboring-star pairs, and edge-Z estimators.

    Input spins have shape ``(..., E)``. Output ``star`` has shape ``(..., V)``;
    ``pair`` and ``z`` have shape ``(..., E)``, with each pair aligned to that
    edge's endpoints. Pair supports are the symmetric difference of the two
    stars, so shared edges cancel. Estimator values can exceed one.
    """
    field_values = _fields(lattice, fields)
    state = np.asarray(spins)
    if state.ndim < 1 or state.shape[-1] != lattice.n_edges or not np.all((state == 1) | (state == -1)):
        raise ValueError("spins must have shape (..., n_edges) with values +1 or -1")
    weighted = state * field_values
    local = np.zeros(state.shape[:-1] + (lattice.n_vertices,), dtype=np.float64)
    for slot in range(lattice.vertex_edge_indices.shape[1]):
        local += weighted[..., lattice.vertex_edge_indices[:, slot]] * lattice.vertex_edge_mask[:, slot]
    endpoints = lattice.edge_vertex_indices
    pair_log = -local[..., endpoints[:, 0]] - local[..., endpoints[:, 1]]
    shared_indices, shared_mask = _shared_incidence(lattice)
    for slot in range(shared_indices.shape[1]):
        pair_log += 2.0 * weighted[..., shared_indices[:, slot]] * shared_mask[:, slot]
    with np.errstate(over="raise", invalid="raise"):
        try:
            star = np.exp(-local)
            pair = np.exp(pair_log)
        except FloatingPointError as error:
            raise ValueError("fields are too large for finite amplitude-ratio estimators") from error
    return {"star": star, "pair": pair, "z": state.astype(np.float64, copy=True)}


def _free_vertices(lattice: Lattice) -> list[int]:
    """Fix one gauge generator per connected component, including isolates."""
    visited: set[int] = set()
    free: list[int] = []
    for root in range(lattice.n_vertices):
        if root in visited:
            continue
        stack = [root]
        visited.add(root)
        while stack:
            vertex = stack.pop()
            if vertex != root:
                free.append(vertex)
            for edge in lattice.vertices[vertex].incident_edges:
                other = edge.vertex_ids[0] if edge.vertex_ids[1] == vertex else edge.vertex_ids[1]
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
    return free


def exact_expectations(lattice: Lattice, fields: Any, *, max_vertices: int = 16) -> dict[str, np.ndarray]:
    """Enumerate the +1 reference star orbit and return exact means.

    ``star``, ``pair``, and ``z`` are expectation arrays. ``states`` and
    ``probabilities`` expose the enumerated normalized target distribution for
    independent validation. A connected lattice has ``2**(V-1)`` states.
    This deliberately rejects larger systems rather than allocate exponentially.
    """
    maximum = _nonnegative_integer(max_vertices, "max_vertices")
    if lattice.n_vertices > maximum:
        raise ValueError("exact enumeration exceeds max_vertices")
    field_values = _fields(lattice, fields)
    free = _free_vertices(lattice)
    count = 1 << len(free)
    pseudospins = np.ones((count, lattice.n_vertices), dtype=np.int8)
    labels = np.arange(count, dtype=np.uint64)
    for bit, vertex in enumerate(free):
        pseudospins[:, vertex] = 1 - 2 * ((labels >> bit) & 1).astype(np.int8)
    endpoints = lattice.edge_vertex_indices
    states = pseudospins[:, endpoints[:, 0]] * pseudospins[:, endpoints[:, 1]]
    log_weights = states @ field_values
    if not np.all(np.isfinite(log_weights)):
        raise ValueError("fields are too large for finite exact log weights")
    weights = np.exp(log_weights - log_weights.max())
    probabilities = weights / weights.sum()
    estimators = local_estimators(lattice, field_values, states)
    result = {name: probabilities @ values for name, values in estimators.items()}
    result.update(states=states, probabilities=probabilities)
    return result
