"""Paired known-field flip estimators and bounded validation summaries.

These helpers do not alter data generation or model inputs. Conditioning over
the two configurations exchanged by an involutive flip gives ``sech(h)`` in
place of ``exp(-h)``. Both require the true simulation fields. The single-draw
variance reduction does not establish an ordering of MCMC asymptotic variance.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from quantum_lattice.lattice import Lattice
from quantum_lattice.observables.statistics import autocorrelation_diagnostic
from quantum_lattice.simulation.solvable import _fields, _nonnegative_integer


@lru_cache(maxsize=16)
def _shared_edges(lattice: Lattice) -> tuple[np.ndarray, np.ndarray]:
    """Compile all common edges, including parallel edges, for each pair."""
    groups: dict[tuple[int, int], list[int]] = {}
    for edge in lattice.edges:
        groups.setdefault(tuple(sorted(edge.vertex_ids)), []).append(edge.id)
    width = max(map(len, groups.values()), default=0)
    indices = np.zeros((lattice.n_edges, width), dtype=np.int64)
    mask = np.zeros_like(indices, dtype=bool)
    for edge in lattice.edges:
        group = groups[tuple(sorted(edge.vertex_ids))]
        indices[edge.id, :len(group)] = group
        mask[edge.id, :len(group)] = True
    return indices, mask


def paired_estimators(lattice: Lattice, fields: Any, states: Any) -> dict:
    """Return ``star/pair -> f/g`` arrays for edge-spin input ``(..., E)``.

    Neighboring pairs are aligned to lattice edge IDs. Their flip support is
    the symmetric difference of endpoint-star supports: every common edge
    cancels, rather than only the particular edge indexing a pair.
    """
    b = _fields(lattice, fields)
    z = np.asarray(states)
    if z.ndim < 1 or z.shape[-1] != lattice.n_edges or not np.all((z == 1) | (z == -1)):
        raise ValueError("states must have shape (..., n_edges) and spins +/-1")
    weighted = z * b
    star_h = np.zeros(z.shape[:-1] + (lattice.n_vertices,), dtype=np.float64)
    for slot in range(lattice.vertex_edge_indices.shape[1]):
        star_h += weighted[..., lattice.vertex_edge_indices[:, slot]] * lattice.vertex_edge_mask[:, slot]
    ends = lattice.edge_vertex_indices
    pair_h = star_h[..., ends[:, 0]] + star_h[..., ends[:, 1]]
    indices, mask = _shared_edges(lattice)
    for slot in range(indices.shape[1]):
        pair_h -= 2 * weighted[..., indices[:, slot]] * mask[:, slot]
    if not np.isfinite(star_h).all() or not np.isfinite(pair_h).all():
        raise ValueError("fields are too large for finite local field sums")
    result = {}
    with np.errstate(over="raise", invalid="raise"):
        try:
            for channel, h in (("star", star_h), ("pair", pair_h)):
                u = np.exp(-np.abs(h))
                result[channel] = {"f": np.exp(-h), "g": 2 * u / (1 + u * u)}
        except FloatingPointError as error:
            raise ValueError("fields are too large for finite paired estimators") from error
    return result


@lru_cache(maxsize=16)
def _orbit(lattice: Lattice) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Enumerate root-fixed pseudospins and each star's label-XOR mask."""
    components: list[list[int]] = []
    visited: set[int] = set()
    for root in range(lattice.n_vertices):
        if root in visited:
            continue
        stack, component = [root], []
        visited.add(root)
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for edge in lattice.vertices[vertex].incident_edges:
                other = edge.vertex_ids[0] if edge.vertex_ids[1] == vertex else edge.vertex_ids[1]
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        components.append(component)
    free = [v for component in components for v in component[1:]]
    labels = np.arange(1 << len(free), dtype=np.int64)
    theta = np.ones((len(labels), lattice.n_vertices), dtype=np.int8)
    masks = [0] * lattice.n_vertices
    for bit, vertex in enumerate(free):
        masks[vertex] = 1 << bit
        theta[:, vertex] = 1 - 2 * ((labels >> bit) & 1)
    for component in components:
        masks[component[0]] = sum(masks[vertex] for vertex in component[1:])
    ends = lattice.edge_vertex_indices
    states = theta[:, ends[:, 0]] * theta[:, ends[:, 1]]
    states.setflags(write=False)
    labels.setflags(write=False)
    return states, labels, tuple(masks)


def exact_reference(lattice: Lattice, fields: Any, *, max_vertices: int = 16) -> dict:
    """Independent amplitude-overlap truth and paired stationary variances.

    One pseudospin is fixed per connected component; isolated stars are the
    identity. The overlap oracle permutes probability amplitudes using star
    labels and never defines truth through either local estimator formula.
    Exponentially large systems are rejected before state allocation.
    """
    maximum = _nonnegative_integer(max_vertices, "max_vertices")
    if lattice.n_vertices > maximum or lattice.n_vertices > 16:
        raise ValueError("exact reference exceeds the 16-vertex enumeration limit")
    b = _fields(lattice, fields)
    states, labels, masks = _orbit(lattice)
    logw = states @ b
    if not np.isfinite(logw).all():
        raise ValueError("nonfinite exact log weights")
    p = np.exp(logw - logw.max())
    p /= p.sum()
    if not np.all(p > 0):
        raise ValueError("exact probabilities underflow; fields exceed numerical range")
    rootp = np.sqrt(p)
    truth = {
        "star": np.asarray([rootp @ rootp[labels ^ mask] for mask in masks]),
        "pair": np.asarray([rootp @ rootp[labels ^ masks[u] ^ masks[v]] for u, v in lattice.edge_vertex_indices]),
    }
    values, variances, max_error = paired_estimators(lattice, b, states), {}, 0.0
    for channel in ("star", "pair"):
        mu = truth[channel]
        f, g = values[channel]["f"], values[channel]["g"]
        vf, vg = p @ ((f - mu) ** 2), p @ ((g - mu) ** 2)
        variances[channel] = {"f": vf, "g": vg}
        checks = (p @ f - mu, p @ g - mu, p @ (f * f) - 1,
                  vf - vg - p @ (1 - g * g), np.maximum(vg - mu * (1 - mu), 0))
        for residual in checks:
            if residual.size:
                max_error = max(max_error, float(np.max(np.abs(residual))))
    return {"states": states, "probabilities": p, "truth": truth,
            "variance": variances, "max_identity_error": max_error}


def _finite_mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size and np.isfinite(values).all() else None


def _block_se(values: np.ndarray, size: int) -> np.ndarray:
    blocks = len(values) // size
    if blocks < 2:
        return np.full(values.shape[1], np.nan)
    means = values[:blocks * size].reshape(blocks, size, values.shape[1]).mean(axis=1)
    return means.std(axis=0, ddof=1) / np.sqrt(blocks)


def summarize_records(lattice: Lattice, fields: Any, states: Any, *, truth: dict | None = None,
                      block_sizes=(45, 90, 150), probe_count: int = 12, seed: Any = 0) -> tuple[dict, dict]:
    """Summarize paired traces without retaining per-record float arrays.

    Summary values are JSON compatible; unavailable SE/variance summaries are
    ``None``. NPZ arrays retain NaNs for unavailable estimates. Half summaries
    use equal first/last ``records//2`` slices, excluding a middle odd record.
    They can be combined across independent chains for conventional split-Rhat.
    ESS probes share coordinate choices between methods and remain descriptive.
    """
    z = np.asarray(states)
    if z.ndim != 2 or len(z) == 0:
        raise ValueError("states must be a nonempty (records, edges) array")
    sizes = tuple(_nonnegative_integer(size, "block size") for size in block_sizes)
    if any(size < 1 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("block sizes must be positive and unique")
    probes = _nonnegative_integer(probe_count, "probe_count")
    values = paired_estimators(lattice, fields, z)
    count, half = len(z), len(z) // 2
    summary: dict = {"records": count, "half_lengths": [half, half], "channels": {},
                     "limitations": "Known-field offline estimators; finite-chain ESS and SE do not certify mixing."}
    arrays: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for channel in ("star", "pair"):
        coordinates = values[channel]["f"].shape[1]
        selected = sorted(rng.choice(coordinates, size=min(probes, coordinates), replace=False).tolist())
        mu = None
        if truth is not None:
            mu = np.asarray(truth[channel], dtype=float)
            if mu.shape != (coordinates,) or not np.isfinite(mu).all():
                raise ValueError("truth must provide finite star/pair coordinate arrays")
            arrays[f"{channel}_truth"] = mu.copy()
        channel_summary: dict = {"coordinates": coordinates}
        for method in ("f", "g"):
            x = values[channel][method]
            prefix = f"{channel}_{method}"
            mean = x.mean(axis=0)
            variance = x.var(axis=0, ddof=1) if count > 1 else np.full(coordinates, np.nan)
            halves = (x[:half], x[-half:]) if half else (x[:0], x[:0])
            half_mean = np.stack([piece.mean(axis=0) if half else np.full(coordinates, np.nan) for piece in halves])
            half_variance = np.stack([piece.var(axis=0, ddof=1) if half > 1 else np.full(coordinates, np.nan) for piece in halves])
            total = x.sum(axis=0)
            shares = np.divide(x.max(axis=0), total, out=np.full(coordinates, np.nan), where=total > 0)
            arrays.update({f"{prefix}_mean": mean, f"{prefix}_variance": variance,
                           f"{prefix}_half_mean": half_mean, f"{prefix}_half_variance": half_variance})
            entry: dict = {"mean_variance": _finite_mean(variance),
                           "mean_half_difference": _finite_mean(np.abs(half_mean[1] - half_mean[0])),
                           "max_record_share": float(np.nanmax(shares)) if shares.size and np.isfinite(shares).any() else None,
                           "fraction_record_share_above_0_1": _finite_mean(shares > .1),
                           "autocorrelation": []}
            if mu is not None:
                error = mean - mu
                entry.update(mse=_finite_mean(error * error), mae=_finite_mean(np.abs(error)),
                             max_absolute_error=float(np.max(np.abs(error))) if coordinates else None)
            for size in sizes:
                se = _block_se(x, size)
                arrays[f"{prefix}_block_se_{size}"] = se
                entry[f"mean_block_se_{size}"] = _finite_mean(se)
                if mu is not None:
                    entry[f"coverage_{size}"] = _finite_mean(np.abs(mean - mu) <= 1.96 * se) if np.isfinite(se).all() else None
            for coordinate in selected:
                entry["autocorrelation"].append({"kind": "fixed-seed random coordinate", "coordinate": coordinate,
                                                **autocorrelation_diagnostic(x[:, coordinate])})
            if coordinates:
                entry["autocorrelation"].append({"kind": "spatial mean", "coordinate": None,
                                                **autocorrelation_diagnostic(x.mean(axis=1))})
            channel_summary[method] = entry
        comparison = {}
        if mu is not None:
            old = np.abs(arrays[f"{channel}_f_mean"] - mu)
            new = np.abs(arrays[f"{channel}_g_mean"] - mu)
            comparison.update(fraction_coordinates_g_absolute_error_worse=_finite_mean(new > old),
                              coordinates_g_absolute_error_worse=int(np.sum(new > old)),
                              coordinates_compared=coordinates)
        channel_summary["comparison"] = comparison
        summary["channels"][channel] = channel_summary
    return summary, arrays


def split_rhat_from_summaries(half_means: Any, half_variances: Any, half_length: int) -> np.ndarray:
    """Conventional split-Rhat from ``(chains, 2, coordinates)`` half arrays.

    This is neither rank-normalized Rhat nor a convergence certificate. A
    zero within-chain variance with nonzero between-chain variance is infinite;
    both zero is undefined (NaN), rather than silently declared converged.
    """
    n = _nonnegative_integer(half_length, "half_length")
    means, variances = np.asarray(half_means, float), np.asarray(half_variances, float)
    if means.ndim < 2 or means.shape != variances.shape or means.shape[1] != 2 or means.shape[0] < 2 or n < 2:
        raise ValueError("need >=2 chains with two equal halves of length >=2")
    within = variances.mean(axis=(0, 1))
    between = n * means.reshape((-1,) + means.shape[2:]).var(axis=0, ddof=1)
    ratio = np.divide((n - 1) / n * within + between / n, within,
                      out=np.full_like(within, np.nan), where=within > 0)
    return np.sqrt(np.where((within == 0) & (between > 0), np.inf, ratio))
