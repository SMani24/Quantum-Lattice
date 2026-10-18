"""Normalized Hamiltonian coefficient distance from the reference paper's D5.

The expansion uses distinct *global* Pauli strings, not a concatenation of
local term lists. Repeated strings are added before taking a norm. Both the
identity and the fixed toric-code stabilizers are included by default.

Only signed simulation fields are accepted here. Assigning signs to a model's
magnitude predictions is a separate inference or explicitly labelled oracle
operation. No field clipping or sign reconstruction occurs in this module.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from quantum_lattice.lattice import Lattice


PauliKey = tuple[str, tuple[int, ...]]
_IDENTITY: PauliKey = ("I", ())
_MODELS = {"z_branch", "two_channel"}
# The method is exponential in each local degree, never in the total qubits.
# Refuse accidentally unbounded local enumeration on an arbitrary graph.
_MAX_LOCAL_DEGREE = 20


def _real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _coefficient_mapping(values: Mapping, name: str) -> dict:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping of coefficients")
    result = {key: _real(value, f"{name} coefficient") for key, value in values.items()}
    if not result or not any(result.values()):
        raise ValueError(f"{name} must contain a nonzero coefficient vector")
    return result


@dataclass(frozen=True)
class PauliExpansion:
    """A canonical expansion with a common finite logarithmic scale.

    Physical coefficients are ``coefficients[P] * exp(log_scale)``. Taking a
    normalized distance requires only the scaled coefficients, so very large
    physical coefficients need not be materialized. The mapping is copied and
    made immutable. Keys are ``('I', ())`` or ``('X'/'Z', sorted_edge_tuple)``.
    """

    n_qubits: int
    coefficients: Mapping[PauliKey, float]
    log_scale: float
    model: str
    include_identity: bool

    def __post_init__(self) -> None:
        if (isinstance(self.n_qubits, (bool, np.bool_))
                or not isinstance(self.n_qubits, (int, np.integer))
                or self.n_qubits < 0):
            raise ValueError("n_qubits must be a nonnegative integer")
        if not isinstance(self.model, str) or self.model not in _MODELS:
            raise ValueError("model must be z_branch or two_channel")
        if not isinstance(self.include_identity, (bool, np.bool_)):
            raise ValueError("include_identity must be boolean")
        values = _coefficient_mapping(self.coefficients, "coefficients")
        for key in values:
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("Pauli keys must be canonical (axis, support) tuples")
            axis, support = key
            if not isinstance(axis, str) or axis not in {"I", "X", "Z"} or not isinstance(support, tuple):
                raise ValueError("Pauli keys must be canonical (axis, support) tuples")
            if any(isinstance(edge, (bool, np.bool_))
                   or not isinstance(edge, (int, np.integer))
                   or edge < 0 or edge >= self.n_qubits for edge in support):
                raise ValueError("Pauli support must contain valid edge IDs")
            if support != tuple(sorted(set(support))):
                raise ValueError("Pauli support must have sorted, distinct edge IDs")
            if (axis == "I") != (not support):
                raise ValueError("only the canonical identity may have empty support")
        if not self.include_identity and values.get(_IDENTITY, 0.0) != 0.0:
            raise ValueError("identity coefficient conflicts with include_identity=False")
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        object.__setattr__(self, "log_scale", _real(self.log_scale, "log_scale"))
        object.__setattr__(self, "include_identity", bool(self.include_identity))
        object.__setattr__(self, "coefficients", MappingProxyType({key: value for key, value in values.items() if value != 0.0}))


def _fields(lattice: Lattice, values: ArrayLike, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.shape != (lattice.n_edges,):
        raise ValueError(f"{name} must contain one finite real value per edge")
    with np.errstate(over="ignore", invalid="ignore"):
        fields = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(fields)):
        raise ValueError(f"{name} must contain one finite real value per edge")
    return fields


def _key(axis: str, support: tuple[int, ...]) -> PauliKey:
    return (axis, support) if support else _IDENTITY


def pauli_coefficients(
    lattice: Lattice,
    b_z: ArrayLike,
    b_x: ArrayLike | None = None,
    *,
    model: str = "two_channel",
    include_identity: bool = True,
) -> PauliExpansion:
    """Expand the Eq6 Z branch or the two-channel Hamiltonian in D7.

    ``z_branch`` is ``-sum(A)-sum(B)+sum_s exp(-sum_s b_z Z)``.
    ``two_channel`` additionally contains ``sum_p exp(-sum_p b_x X)``;
    omitted ``b_x`` means a zero field, whose exponential sum still adds
    ``n_faces * I``. These conventions differ by an energy shift at zero X
    field. Setting ``include_identity=False`` removes the *combined* identity
    coefficient and compares centered Hamiltonians.

    For each support G, the subset coefficient is a product of cosh factors
    times selected -tanh factors. A common logarithmic cosh scale prevents
    overflow without changing a normalized coefficient vector. Runtime is
    proportional to the number of local subsets for bounded-degree topology.
    """
    if not isinstance(lattice, Lattice):
        raise ValueError("lattice must be a Lattice")
    if not isinstance(model, str) or model not in _MODELS:
        raise ValueError("model must be z_branch or two_channel")
    if not isinstance(include_identity, (bool, np.bool_)):
        raise ValueError("include_identity must be boolean")
    fields_z = _fields(lattice, b_z, "b_z")
    fields_x = np.zeros(lattice.n_edges) if b_x is None else _fields(lattice, b_x, "b_x")
    if model == "z_branch" and np.any(fields_x != 0.0):
        raise ValueError("z_branch does not contain nonzero b_x fields")

    stars = [tuple(sorted(vertex.edge_ids)) for vertex in lattice.vertices]
    faces = [tuple(sorted(face.edge_ids)) for face in lattice.faces]
    groups = [("Z", support, fields_z) for support in stars]
    if model == "two_channel":
        groups.extend(("X", support, fields_x) for support in faces)
    if any(len(support) > _MAX_LOCAL_DEGREE for _, support, _ in groups):
        raise ValueError(f"local exponential support exceeds {_MAX_LOCAL_DEGREE} edges")

    log_cosh = []
    try:
        for _, support, fields in groups:
            log_cosh.append(math.fsum(
                abs(float(fields[edge]))
                + math.log1p(math.exp(-2.0 * abs(float(fields[edge]))))
                - math.log(2.0)
                for edge in support
            ))
    except OverflowError as error:
        raise ValueError("fields are too large for a finite logarithmic scale") from error
    if not all(math.isfinite(value) for value in log_cosh):
        raise ValueError("fields are too large for a finite logarithmic scale")
    scale = max([0.0, *log_cosh])
    coefficients: dict[PauliKey, float] = {}

    def add(key: PauliKey, value: float) -> None:
        coefficients[key] = coefficients.get(key, 0.0) + value

    fixed = -math.exp(-scale)
    for support in stars:
        add(_key("X", support), fixed)
    for support in faces:
        add(_key("Z", support), fixed)
    for (axis, support, fields), logarithm in zip(groups, log_cosh):
        terms: list[tuple[tuple[int, ...], float]] = [((), math.exp(logarithm - scale))]
        for edge in support:
            factor = -math.tanh(float(fields[edge]))
            if factor != 0.0:
                terms.extend([(subset + (edge,), value * factor) for subset, value in terms])
        for subset, value in terms:
            add(_key(axis, subset), value)
    if not include_identity:
        coefficients.pop(_IDENTITY, None)
    return PauliExpansion(lattice.n_edges, coefficients, scale, model, bool(include_identity))


def _normalized(values: Mapping, name: str) -> dict:
    coefficients = _coefficient_mapping(values, name)
    maximum = max(abs(value) for value in coefficients.values())
    scaled = {key: value / maximum for key, value in coefficients.items()}
    norm = math.sqrt(math.fsum(value * value for value in scaled.values()))
    return {key: value / norm for key, value in scaled.items()}


def normalized_coefficient_distance(
    left: Mapping[Hashable, Real], right: Mapping[Hashable, Real]
) -> float:
    """Euclidean distance between normalized nonzero coefficient mappings.

    Missing keys mean zero; mappings must refer to the same coefficient basis.
    A positive global scale cancels, a negative scale reverses the vector, and
    a zero vector has no normalized distance. Max-absolute rescaling handles
    arbitrarily small or large finite input magnitudes safely.
    """
    first = _normalized(left, "left")
    second = _normalized(right, "right")
    squared = math.fsum((first.get(key, 0.0) - second.get(key, 0.0)) ** 2
                        for key in first.keys() | second.keys())
    # The exact range is [0,2]; remove only possible roundoff beyond its bound.
    return min(2.0, math.sqrt(squared))


def normalized_pauli_distance(left: PauliExpansion, right: PauliExpansion) -> float:
    """Evaluate D5 after checking matching Hilbert-space/model conventions."""
    if not isinstance(left, PauliExpansion) or not isinstance(right, PauliExpansion):
        raise ValueError("left and right must be PauliExpansion instances")
    if (left.n_qubits, left.model, left.include_identity) != (right.n_qubits, right.model, right.include_identity):
        raise ValueError("expansions must use the same qubits, model and identity convention")
    return normalized_coefficient_distance(left.coefficients, right.coefficients)


def hamiltonian_error(
    lattice: Lattice,
    true_b_z: ArrayLike,
    recovered_b_z: ArrayLike,
    true_b_x: ArrayLike | None = None,
    recovered_b_x: ArrayLike | None = None,
    *,
    model: str = "two_channel",
    include_identity: bool = True,
) -> float:
    """D5 for signed true/recovered fields with explicit model conventions."""
    true = pauli_coefficients(lattice, true_b_z, true_b_x, model=model, include_identity=include_identity)
    recovered = pauli_coefficients(lattice, recovered_b_z, recovered_b_x, model=model, include_identity=include_identity)
    return normalized_pauli_distance(true, recovered)
