"""Immutable edge-qubit topology with array-based stabilizer calculations.

Vertices, edges, and faces describe connectivity once. Error/spin samples are
separate arrays indexed by edge ID; topology objects never hold sample flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _readonly(values: ArrayLike, dtype: object) -> NDArray:
    """Use immutable backing bytes, so writeability cannot be re-enabled."""
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _compile_incidence(groups: Sequence[Sequence[int]]) -> tuple[NDArray, NDArray]:
    width = max((len(group) for group in groups), default=0)
    indices = np.zeros((len(groups), width), dtype=np.int64)
    mask = np.zeros(indices.shape, dtype=bool)
    for row, group in enumerate(groups):
        indices[row, : len(group)] = group
        mask[row, : len(group)] = True
    return _readonly(indices, np.int64), _readonly(mask, bool)


@dataclass(frozen=True, slots=True)
class Vertex:
    """A fixed vertex; incident edges resolve through its owning lattice."""

    id: int
    position: tuple[float, ...] | None
    edge_ids: tuple[int, ...]
    _lattice: Lattice = field(repr=False)

    @property
    def incident_edges(self) -> tuple[Edge, ...]:
        return tuple(self._lattice.edges[index] for index in self.edge_ids)


@dataclass(frozen=True, slots=True)
class Edge:
    """An undirected edge qubit with two distinct endpoint vertices."""

    id: int
    vertex_ids: tuple[int, int]
    face_ids: tuple[int, ...]
    _lattice: Lattice = field(repr=False)

    @property
    def endpoints(self) -> tuple[Vertex, Vertex]:
        return tuple(self._lattice.vertices[index] for index in self.vertex_ids)

    @property
    def incident_faces(self) -> tuple[Face, ...]:
        return tuple(self._lattice.faces[index] for index in self.face_ids)


@dataclass(frozen=True, slots=True)
class Face:
    """An ordered closed boundary of edge qubits (a plaquette)."""

    id: int
    edge_ids: tuple[int, ...]
    _lattice: Lattice = field(repr=False)

    @property
    def boundary_edges(self) -> tuple[Edge, ...]:
        return tuple(self._lattice.edges[index] for index in self.edge_ids)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Lattice:
    """Own immutable topology objects and compiled numerical connectivity.

    IDs are the positions in the supplied sequences. ``edge_endpoints[e]`` is
    the pair of vertex IDs for edge e; ``face_boundaries[f]`` is an ordered
    closed trail of distinct edge IDs for face f. A boundary may revisit a
    vertex, but not an edge. Parallel edges are allowed, but self-loops are
    rejected. Faces must meet stars on an even number of edges.

    ``periodic``/``closed`` require exactly two faces incident on every edge.
    ``open``/``graph`` permit fewer, including a graph without faces. These
    boundary labels describe incidence validation rather than create geometry.

    Padded vertex/face edge-index arrays and their masks are read-only. The
    objects are convenient for inspection, while syndromes use these arrays
    in vectorized batches without Python loops over samples or vertices.
    """

    vertices: tuple[Vertex, ...]
    edges: tuple[Edge, ...]
    faces: tuple[Face, ...]
    geometry: str
    boundary: str
    shape: tuple[int, ...] | None
    edge_vertex_indices: NDArray[np.int64] = field(repr=False)
    vertex_degrees: NDArray[np.int64] = field(repr=False)
    face_degrees: NDArray[np.int64] = field(repr=False)
    vertex_edge_indices: NDArray[np.int64] = field(repr=False)
    vertex_edge_mask: NDArray[np.bool_] = field(repr=False)
    face_edge_indices: NDArray[np.int64] = field(repr=False)
    face_edge_mask: NDArray[np.bool_] = field(repr=False)

    def __init__(
        self,
        n_vertices: int,
        edge_endpoints: ArrayLike,
        face_boundaries: Iterable[Iterable[int]] = (),
        *,
        positions: Sequence[Sequence[float] | None] | None = None,
        geometry: str = "generic",
        boundary: str = "graph",
        shape: Sequence[int] | None = None,
    ) -> None:
        count = _integer(n_vertices, "n_vertices")
        if count < 1:
            raise ValueError("n_vertices must be positive")
        if boundary not in {"periodic", "closed", "open", "graph"}:
            raise ValueError("boundary must be periodic, closed, open, or graph")
        if not isinstance(geometry, str) or not geometry:
            raise ValueError("geometry must be a nonempty string")

        endpoints = np.asarray(edge_endpoints)
        if endpoints.size == 0 and endpoints.shape == (0,):
            endpoints = np.empty((0, 2), dtype=np.int64)
        if endpoints.ndim != 2 or endpoints.shape[1] != 2:
            raise ValueError("edge_endpoints must have shape (n_edges, 2)")
        if endpoints.dtype.kind not in "iu":
            raise ValueError("edge endpoint IDs must be integers")
        if np.any(endpoints < 0) or np.any(endpoints >= count):
            raise ValueError("edge endpoint IDs must be valid vertex IDs")
        if np.any(endpoints[:, 0] == endpoints[:, 1]):
            raise ValueError("self-loop edges are not supported")
        endpoints = np.asarray(endpoints, dtype=np.int64)

        vertex_edges: list[list[int]] = [[] for _ in range(count)]
        for edge_id, (left, right) in enumerate(endpoints):
            vertex_edges[int(left)].append(edge_id)
            vertex_edges[int(right)].append(edge_id)

        boundaries: list[tuple[int, ...]] = []
        edge_faces: list[list[int]] = [[] for _ in range(len(endpoints))]
        for face_id, raw_boundary in enumerate(face_boundaries):
            edges = tuple(_integer(edge, "face edge IDs") for edge in raw_boundary)
            if len(edges) < 3:
                raise ValueError("each face must have at least three boundary edges")
            if len(set(edges)) != len(edges):
                raise ValueError("a face boundary must not repeat an edge")
            if any(edge < 0 or edge >= len(endpoints) for edge in edges):
                raise ValueError("face boundaries must contain valid edge IDs")
            if not self._is_closed_boundary(edges, endpoints):
                raise ValueError("face boundary edges must form an ordered closed trail")
            intersections = np.bincount(endpoints[list(edges)].ravel(), minlength=count)
            if np.any(intersections % 2):
                raise ValueError("star and plaquette supports must intersect evenly")
            for edge in edges:
                edge_faces[edge].append(face_id)
            boundaries.append(edges)

        if any(len(faces) > 2 for faces in edge_faces):
            raise ValueError("an edge cannot be incident on more than two faces")
        if boundary in {"periodic", "closed"} and any(
            len(faces) != 2 for faces in edge_faces
        ):
            raise ValueError("closed/periodic topology requires two faces per edge")
        if boundary in {"periodic", "closed"} and (
            not boundaries or any(not edges for edges in vertex_edges)
        ):
            raise ValueError("closed/periodic topology requires faces and no isolated vertices")

        if positions is None:
            coordinates = (None,) * count
        else:
            if len(positions) != count:
                raise ValueError("positions must contain one entry per vertex")
            converted: list[tuple[float, ...] | None] = []
            for position in positions:
                if position is None:
                    converted.append(None)
                    continue
                coordinate = tuple(float(value) for value in position)
                if not coordinate or not np.all(np.isfinite(coordinate)):
                    raise ValueError("vertex positions must have finite coordinates")
                converted.append(coordinate)
            coordinates = tuple(converted)

        lattice_shape = None
        if shape is not None:
            lattice_shape = tuple(_integer(value, "shape entries") for value in shape)
            if not lattice_shape or any(value < 1 for value in lattice_shape):
                raise ValueError("shape must contain positive integers")

        vertex_indices, vertex_mask = _compile_incidence(vertex_edges)
        face_indices, face_mask = _compile_incidence(boundaries)
        objects = {
            "geometry": geometry,
            "boundary": boundary,
            "shape": lattice_shape,
            "edge_vertex_indices": _readonly(endpoints, np.int64),
            "vertex_degrees": _readonly([len(edges) for edges in vertex_edges], np.int64),
            "face_degrees": _readonly([len(edges) for edges in boundaries], np.int64),
            "vertex_edge_indices": vertex_indices,
            "vertex_edge_mask": vertex_mask,
            "face_edge_indices": face_indices,
            "face_edge_mask": face_mask,
            "vertices": tuple(
                Vertex(index, coordinates[index], tuple(edges), self)
                for index, edges in enumerate(vertex_edges)
            ),
            "edges": tuple(
                Edge(index, (int(pair[0]), int(pair[1])), tuple(edge_faces[index]), self)
                for index, pair in enumerate(endpoints)
            ),
            "faces": tuple(
                Face(index, edges, self) for index, edges in enumerate(boundaries)
            ),
        }
        for name, value in objects.items():
            object.__setattr__(self, name, value)

    @staticmethod
    def _is_closed_boundary(edges: tuple[int, ...], endpoints: NDArray) -> bool:
        first_left, first_right = endpoints[edges[0]]
        for start, current in ((first_left, first_right), (first_right, first_left)):
            for edge in edges[1:]:
                left, right = endpoints[edge]
                if left == current:
                    current = right
                elif right == current:
                    current = left
                else:
                    break
            else:
                if current == start:
                    return True
        return False

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def _binary_errors(self, errors: ArrayLike) -> NDArray[np.bool_]:
        array = np.asarray(errors)
        if array.ndim < 1 or array.shape[-1] != self.n_edges:
            raise ValueError(f"errors must have shape (..., {self.n_edges})")
        if array.dtype != np.bool_:
            if array.dtype.kind not in "iuf" or not np.all((array == 0) | (array == 1)):
                raise ValueError("errors must contain only binary values")
            array = array.astype(bool)
        return array

    def _syndrome(self, errors: ArrayLike, indices: NDArray, mask: NDArray) -> NDArray:
        flags = self._binary_errors(errors)
        syndrome = np.zeros(flags.shape[:-1] + (indices.shape[0],), dtype=bool)
        # Loop over the small incidence width, never over patterns or vertices.
        # This avoids a large (..., vertices, width) gather and reduction;
        # masks keep the same calculation valid for irregular degrees.
        for slot in range(indices.shape[1]):
            syndrome ^= np.take(flags, indices[:, slot], axis=-1) & mask[:, slot]
        return syndrome

    def star_syndrome(self, errors: ArrayLike) -> NDArray[np.bool_]:
        """Return odd vertex-edge parity for flags of shape (..., n_edges)."""
        return self._syndrome(errors, self.vertex_edge_indices, self.vertex_edge_mask)

    def plaquette_syndrome(self, errors: ArrayLike) -> NDArray[np.bool_]:
        """Return face-edge parity (for X-error flags), shape (..., n_faces).

        This is a separate channel: Z errors cause star defects and X errors
        cause plaquette defects. The parity calculation alone does not assign
        a physical Pauli type to its input flags.
        """
        return self._syndrome(errors, self.face_edge_indices, self.face_edge_mask)

    def defect_fractions(self, errors: ArrayLike) -> NDArray[np.float64]:
        """Average star-defect fraction for each whole edge-error pattern."""
        return self.star_syndrome(errors).mean(axis=-1, dtype=np.float64)
