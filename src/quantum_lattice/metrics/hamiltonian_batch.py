"""Compiled NumPy batches of the paper's normalized Hamiltonian distance.

This is the same global Pauli expansion as :mod:`hamiltonian`, with topology
and subset-to-Pauli indices compiled once. It never enumerates Hilbert-space
states or calls the scalar metric once per Hamiltonian. Inputs are signed
fields; choosing signs for a magnitude prediction remains the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from quantum_lattice.lattice import Lattice


_MAX_LOCAL_DEGREE = 20
_MAX_COMPILED_TERMS = 2_000_000
_IDENTITY = ("I", ())


@dataclass(frozen=True, slots=True)
class _Groups:
    axis: str
    edges: np.ndarray
    destinations: np.ndarray


def _readonly(values, dtype):
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class CompiledHamiltonianMetric:
    """Compile a fixed lattice and compare signed field arrays ``[H, E]``.

    ``distances`` returns separate unsquared unit-vector distances including
    the combined identity coefficient (``full``), and after removing it
    (``traceless``). The default two-channel Hamiltonian includes the zero-X
    exponential sum when X fields are omitted. This convention is an energy
    shift relative to ``z_branch`` and agrees with the scalar implementation.

    All duplicate global Pauli strings, including collisions with fixed
    stabilizers or the identity, are added *before* computing either norm.
    Memory scales with the requested batch and bounded-degree local subsets;
    callers should chunk large datasets. Compilation refuses local degrees
    above 20 or more than two million enumerated local subset contributions.
    """

    def __init__(self, lattice: Lattice, *, model: str = "two_channel"):
        if not isinstance(lattice, Lattice):
            raise ValueError("lattice must be a Lattice")
        if not isinstance(model, str) or model not in {"z_branch", "two_channel"}:
            raise ValueError("model must be z_branch or two_channel")
        stars = [tuple(sorted(vertex.edge_ids)) for vertex in lattice.vertices]
        faces = [tuple(sorted(face.edge_ids)) for face in lattice.faces]
        groups = [("Z", support) for support in stars]
        if model == "two_channel":
            groups += [("X", support) for support in faces]
        if any(len(support) > _MAX_LOCAL_DEGREE for _, support in groups):
            raise ValueError(f"local exponential support exceeds {_MAX_LOCAL_DEGREE} edges")
        if sum(1 << len(support) for _, support in groups) > _MAX_COMPILED_TERMS:
            raise ValueError("compiled local subset contributions exceed safe enumeration limit")

        keys = {_IDENTITY: 0}

        def index(axis, support):
            key = (axis, support) if support else _IDENTITY
            return keys.setdefault(key, len(keys))

        fixed = [index("X", support) for support in stars]
        fixed += [index("Z", support) for support in faces]
        compiled = []
        # Grouping equal degrees permits vectorized subset products across
        # Hamiltonians and local supports without padding fictitious edges.
        for axis, degree in sorted({(axis, len(support)) for axis, support in groups}):
            supports = [support for group_axis, support in groups
                        if group_axis == axis and len(support) == degree]
            destinations = []
            for support in supports:
                subsets = [()]
                for edge in support:
                    subsets += [subset + (edge,) for subset in subsets]
                destinations.append([index(axis, subset) for subset in subsets])
            compiled.append(_Groups(
                axis,
                _readonly(np.asarray(supports).reshape(len(supports), degree), np.int64),
                _readonly(destinations, np.int64),
            ))
        self.lattice = lattice
        self.model = model
        self.n_qubits = lattice.n_edges
        self.n_coefficients = len(keys)
        self.pauli_keys = tuple(keys)
        self._groups = tuple(compiled)
        self._fixed = _readonly(np.bincount(fixed, minlength=len(keys)), np.float64)

    def _fields(self, values: ArrayLike, name: str, shape=None):
        raw = np.asarray(values)
        if (raw.dtype.kind not in "iuf" or raw.ndim != 2
                or raw.shape[0] < 1 or raw.shape[1] != self.n_qubits
                or (shape is not None and raw.shape != shape)):
            raise ValueError(f"{name} must be a nonempty finite real array [H, {self.n_qubits}] with matching shape")
        with np.errstate(over="ignore", invalid="ignore"):
            fields = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(fields)):
            raise ValueError(f"{name} must contain finite real fields")
        return fields

    def _coefficients(self, z, x):
        logarithms = []
        factors = []
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for group in self._groups:
                local = (z if group.axis == "Z" else x)[:, group.edges]
                absolute = np.abs(local)
                log_cosh = absolute + np.log1p(np.exp(-2.0 * absolute)) - np.log(2.0)
                logarithms.append(np.sum(log_cosh, axis=-1))
                factors.append(-np.tanh(local))
        if any(not np.all(np.isfinite(values)) for values in logarithms):
            raise ValueError("fields are too large for a finite logarithmic scale")
        scale = np.zeros(len(z), dtype=np.float64)
        for values in logarithms:
            scale = np.maximum(scale, np.max(values, axis=1))
        coefficients = -np.exp(-scale[:, None]) * self._fixed[None, :]
        for group, logarithm, factor in zip(self._groups, logarithms, factors):
            terms = np.exp(logarithm - scale[:, None])[..., None]
            for edge in range(factor.shape[-1]):
                terms = np.concatenate((terms, terms * factor[..., edge, None]), axis=-1)
            # A repeated destination is a repeated GLOBAL Pauli string, not
            # an extra orthogonal coordinate. add.at performs the required sum.
            np.add.at(coefficients.T, group.destinations.ravel(),
                      terms.reshape(len(z), -1).T)
        return coefficients

    @staticmethod
    def _normalize(coefficients):
        maximum = np.max(np.abs(coefficients), axis=1)
        if np.any(maximum == 0.0) or not np.all(np.isfinite(maximum)):
            raise ValueError("Hamiltonian must contain a nonzero finite coefficient vector")
        scaled = coefficients / maximum[:, None]
        return scaled / np.sqrt(np.sum(scaled * scaled, axis=1))[:, None]

    def distances(self, true_b_z: ArrayLike, recovered_b_z: ArrayLike,
                  true_b_x: ArrayLike | None = None,
                  recovered_b_x: ArrayLike | None = None) -> dict[str, np.ndarray]:
        """Return float64 arrays ``full`` and ``traceless``, one score per H.

        The final step subtracts normalized vectors directly instead of using
        ``sqrt(2-2*cosine)``, avoiding cancellation for nearly equal fields.
        No field clipping, absolute value, or sign reconstruction is applied.
        """
        true_z = self._fields(true_b_z, "true_b_z")
        recovered_z = self._fields(recovered_b_z, "recovered_b_z", true_z.shape)
        true_x = np.zeros_like(true_z) if true_b_x is None else self._fields(true_b_x, "true_b_x", true_z.shape)
        recovered_x = np.zeros_like(true_z) if recovered_b_x is None else self._fields(recovered_b_x, "recovered_b_x", true_z.shape)
        if self.model == "z_branch" and (np.any(true_x != 0) or np.any(recovered_x != 0)):
            raise ValueError("z_branch does not contain nonzero b_x fields")
        left = self._coefficients(true_z, true_x)
        right = self._coefficients(recovered_z, recovered_x)
        result = {}
        for name, start in (("full", 0), ("traceless", 1)):
            # The identity is canonical coordinate zero. Remove the combined
            # coefficient, never individual local identity contributions.
            difference = self._normalize(left[:, start:]) - self._normalize(right[:, start:])
            result[name] = np.minimum(2.0, np.sqrt(np.sum(difference * difference, axis=1)))
        return result
