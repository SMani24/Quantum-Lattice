"""Lossless periodic-square observable maps from existing all-edge MLP rows.

This is a layout change, not a new estimator or extra measurement. Each star
expectation occurs at several edge endpoints in the source rows. Those copies
must agree exactly: independent-chain appearances cannot be silently averaged
into a map. Inputs may exceed one (the original f estimator is not clipped).

The map channels are A[y,x], AA_horizontal[y,x], AA_vertical[y,x]. Targets have
horizontal then vertical channels. Edge e=2*(y*L+x)+orientation leaves (y,x)
toward x+1 (horizontal, orientation 0) or y+1 (vertical, orientation 1), modulo L.
The inverse reconstructs every original local triple and original row order.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantum_lattice.data.mlp_data import EdgeRows, LearningData
from quantum_lattice.lattice.square import validate_side_length

INPUT_CHANNELS = ("mean_A", "mean_Au_Av_horizontal", "mean_Au_Av_vertical")
TARGET_CHANNELS = ("abs_b_z_horizontal", "abs_b_z_vertical")
MAP_VERSION = "square-observable-map-v1"


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _numeric(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite real numeric array of shape {shape}")
    return array


def _identities(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "US" or np.any(np.char.str_len(array) == 0):
        raise ValueError(f"{label} must contain nonempty string identities with shape {shape}")
    return array


@dataclass(frozen=True)
class ObservableMaps:
    """One split of whole-Hamiltonian maps, preserving original edge-row order.

    ``source_row_indices[n,e]`` is the row index in the source EdgeRows for
    Hamiltonian n and canonical edge e. Hamiltonian order is first appearance
    in those source rows. Arrays own read-only copies; no input is aliased.
    """

    x: np.ndarray
    y: np.ndarray
    hamiltonian_ids: np.ndarray
    hamiltonian_uids: np.ndarray
    source_row_indices: np.ndarray
    L: int

    def __post_init__(self):
        length = validate_side_length(self.L)
        count = len(self.hamiltonian_ids)
        if count == 0:
            raise ValueError("observable maps need at least one Hamiltonian")
        x = _numeric(self.x, (count, 3, length, length), "observable inputs")
        y = _numeric(self.y, (count, 2, length, length), "edge targets")
        if np.any(y < 0):
            raise ValueError("edge targets must be nonnegative field magnitudes")
        names = _identities(self.hamiltonian_ids, (count,), "Hamiltonian ids")
        uids = _identities(self.hamiltonian_uids, (count,), "Hamiltonian UIDs")
        if len(np.unique(names)) != count or len(np.unique(uids)) != count:
            raise ValueError("each map must represent a unique Hamiltonian id and UID")
        edges = 2 * length * length
        order = np.asarray(self.source_row_indices)
        if (order.shape != (count, edges) or order.dtype.kind not in "iu"
                or not np.array_equal(np.sort(order, axis=None), np.arange(count * edges))):
            raise ValueError("source_row_indices must be a permutation of all source rows")
        for name, value in (("x", x), ("y", y), ("hamiltonian_ids", names),
                            ("hamiltonian_uids", uids), ("source_row_indices", order)):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "L", length)

    def __len__(self):
        return len(self.x)


@dataclass(frozen=True)
class ObservableMapData:
    """Train/validation maps with the source eligibility and provenance intact."""

    train: ObservableMaps
    validation: ObservableMaps
    test: None
    L: int
    provenance: dict
    dataset_dir: Path


def edge_rows_to_maps(rows: EdgeRows, L: int) -> ObservableMaps:
    """Repack complete edge rows, rejecting lost information or duplicate stars.

    Arbitrarily interleaved/shuffled rows are accepted. Each Hamiltonian needs
    exactly one row for every canonical edge; repeated ids must have one UID,
    and one UID cannot appear under multiple ids. No sorting assumption is made.
    """
    length = validate_side_length(L)
    edges = 2 * length * length
    count_rows = len(rows.x)
    if count_rows == 0 or count_rows % edges:
        raise ValueError("all_edges maps require complete nonempty Hamiltonians")
    x = _numeric(rows.x, (count_rows, 3), "local observable inputs")
    y = _numeric(rows.y, (count_rows, 1), "local magnitude targets")
    names = _identities(rows.hamiltonian_ids, (count_rows,), "Hamiltonian ids")
    uids = _identities(rows.hamiltonian_uids, (count_rows,), "Hamiltonian UIDs")
    edge_ids = np.asarray(rows.edge_ids)
    if (edge_ids.shape != (count_rows,) or edge_ids.dtype.kind not in "iu"
            or np.any(edge_ids < 0) or np.any(edge_ids >= edges)):
        raise ValueError("edge_ids must be canonical integer square-lattice edges")
    edge_ids = edge_ids.astype(np.int64, copy=False)

    unique, first, inverse, counts = np.unique(names, return_index=True, return_inverse=True, return_counts=True)
    if np.any(counts != edges):
        raise ValueError("every Hamiltonian must contain exactly all 2*L*L edges")
    first_order = np.argsort(first)
    sorted_to_first = np.empty(len(first_order), dtype=np.int64)
    sorted_to_first[first_order] = np.arange(len(first_order))
    group = sorted_to_first[inverse]
    first = first[first_order]
    unique = unique[first_order]
    group_uids = uids[first]
    if not np.all(uids == group_uids[group]):
        raise ValueError("a Hamiltonian id has inconsistent UIDs across its edges")
    if len(np.unique(group_uids)) != len(group_uids):
        raise ValueError("one Hamiltonian UID appears under multiple ids")

    slots = group * edges + edge_ids
    if len(np.unique(slots)) != count_rows:
        raise ValueError("duplicate edge row or missing edge within a Hamiltonian")
    source_indices = np.empty(count_rows, dtype=np.int64)
    source_indices[slots] = np.arange(count_rows)
    source_indices = source_indices.reshape(-1, edges)
    canonical_x = x[source_indices]
    # Every vertex has one outgoing horizontal edge. Its first feature supplies
    # a candidate star; all remaining appearances must match it exactly.
    stars = canonical_x[:, 0::2, 0]
    starts = np.arange(edges) // 2
    ys, xs = np.divmod(starts, length)
    ends = np.where(np.arange(edges) % 2 == 0,
                    ys * length + (xs + 1) % length,
                    ((ys + 1) % length) * length + xs)
    for column, endpoints in ((0, starts), (1, ends)):
        matches = canonical_x[:, :, column] == stars[:, endpoints]
        if not matches.all():
            n, edge = np.argwhere(~matches)[0]
            raise ValueError(
                f"inconsistent duplicate star expectation for Hamiltonian {unique[n]!r}, "
                f"vertex {endpoints[edge]}, edge {edge}, local column {column}; "
                "a lossless shared-star map cannot average independent appearances"
            )
    count = len(unique)
    inputs = np.empty((count, 3, length, length), dtype=x.dtype)
    inputs[:, 0] = stars.reshape(count, length, length)
    inputs[:, 1:] = canonical_x[:, :, 2].reshape(count, length, length, 2).transpose(0, 3, 1, 2)
    targets = y[source_indices, 0].reshape(count, length, length, 2).transpose(0, 3, 1, 2)
    return ObservableMaps(inputs, targets, unique, group_uids, source_indices, length)


def flatten_edge_maps(maps: ObservableMaps, values: np.ndarray) -> np.ndarray:
    """Return edge outputs (N,2,L,L) as (original row count,1), without clipping.

    Useful for comparing CNN predictions to the existing row-based MLP/metrics.
    Finite signed predictions are accepted even though training targets are
    nonnegative. Output is a separate read-only array in original row order.
    """
    array = _numeric(values, (len(maps), 2, maps.L, maps.L), "edge-map values")
    canonical = array.transpose(0, 2, 3, 1).reshape(-1)
    result = np.empty((canonical.size, 1), dtype=array.dtype)
    result[maps.source_row_indices.reshape(-1), 0] = canonical
    return _readonly(result)


def maps_to_edge_rows(maps: ObservableMaps) -> EdgeRows:
    """Recover all local triples, targets, identities and original row ordering."""
    count, length = len(maps), maps.L
    edges = 2 * length * length
    star, horizontal, vertical = maps.x[:, 0], maps.x[:, 1], maps.x[:, 2]
    canonical = np.empty((count, length, length, 2, 3), dtype=maps.x.dtype)
    canonical[:, :, :, 0, 0] = star
    canonical[:, :, :, 1, 0] = star
    canonical[:, :, :, 0, 1] = np.roll(star, -1, axis=2)
    canonical[:, :, :, 1, 1] = np.roll(star, -1, axis=1)
    canonical[:, :, :, 0, 2] = horizontal
    canonical[:, :, :, 1, 2] = vertical
    positions = maps.source_row_indices.reshape(-1)
    local = np.empty((count * edges, 3), dtype=maps.x.dtype)
    local[positions] = canonical.reshape(-1, 3)
    names = np.empty(count * edges, dtype=maps.hamiltonian_ids.dtype)
    uids = np.empty(count * edges, dtype=maps.hamiltonian_uids.dtype)
    edge_ids = np.empty(count * edges, dtype=np.int64)
    names[positions] = np.repeat(maps.hamiltonian_ids, edges)
    uids[positions] = np.repeat(maps.hamiltonian_uids, edges)
    edge_ids[positions] = np.tile(np.arange(edges), count)
    return EdgeRows(_readonly(local), flatten_edge_maps(maps, maps.y),
                    _readonly(names), _readonly(uids), _readonly(edge_ids))


def map_learning_data(data: LearningData) -> ObservableMapData:
    """Adapt non-test all-edge splits without changing group membership.

    This function performs no disk access. Loaded test rows are rejected rather
    than silently discarded, and train/validation Hamiltonian ids and UIDs must
    both be disjoint. Exact-view ``training_eligible=False`` provenance remains
    unchanged; map conversion does not authorize training on that view.
    """
    if data.mode != "all_edges":
        raise ValueError("observable maps require mode='all_edges'")
    if data.test is not None:
        raise ValueError("observable-map development accepts no loaded test split")
    train = edge_rows_to_maps(data.train, data.L)
    validation = edge_rows_to_maps(data.validation, data.L)
    for attribute in ("hamiltonian_ids", "hamiltonian_uids"):
        if np.intersect1d(getattr(train, attribute), getattr(validation, attribute)).size:
            raise ValueError(f"train/validation leakage through {attribute}")
    provenance = deepcopy(data.provenance)
    provenance["observable_map"] = {
        "version": MAP_VERSION, "input_channels": list(INPUT_CHANNELS),
        "target_channels": list(TARGET_CHANNELS), "boundary": "periodic",
        "geometry": "square", "source_information": "identical all-edge local triples",
        "duplicate_star_policy": "exact equality required; never averaged",
        "test_payload_loaded": False,
    }
    return ObservableMapData(train, validation, None, train.L, provenance, data.dataset_dir)
