"""Periodic square-lattice incidence for edge-qubit phase errors.

The last three axes of an error array are ``(row, column, orientation)``.
Orientation 0 is the horizontal edge from (y, x) to (y, x + 1);
orientation 1 is the vertical edge from (y, x) to (y + 1, x).
All coordinates are periodic. True/1 denotes an applied Pauli Z error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_side_length(side_length: int) -> int:
    """Return a valid integer side length, excluding degenerate tiny tori."""
    if isinstance(side_length, (bool, np.bool_)) or not isinstance(
        side_length, (int, np.integer)
    ):
        raise ValueError("side_length must be an integer of at least 3")
    if side_length < 3:
        raise ValueError("side_length must be at least 3")
    return int(side_length)


def star_syndrome(errors: ArrayLike) -> NDArray[np.bool_]:
    """Return star defects for one pattern or a batch of edge-error patterns.

    A negative star outcome corresponds to odd parity of its four incident
    phase errors. The returned shape is ``errors.shape[:-1]``. This is an
    error-pattern calculation, not ground-state Metropolis sampling.
    """
    errors = np.asarray(errors)
    if errors.ndim < 3 or errors.shape[-1] != 2:
        raise ValueError("errors must have shape (..., L, L, 2)")
    if errors.shape[-3] != errors.shape[-2]:
        raise ValueError("the row and column lattice dimensions must match")
    validate_side_length(errors.shape[-2])
    if errors.dtype != np.bool_:
        if not np.all((errors == 0) | (errors == 1)):
            raise ValueError("errors must contain only binary values")
        errors = errors.astype(bool)
    horizontal = errors[..., 0]
    vertical = errors[..., 1]
    return (
        horizontal
        ^ np.roll(horizontal, 1, axis=-1)
        ^ vertical
        ^ np.roll(vertical, 1, axis=-2)
    )


def defect_fractions(errors: ArrayLike) -> NDArray[np.float64]:
    """Return the fraction of defective vertices for each whole pattern."""
    return star_syndrome(errors).mean(axis=(-2, -1), dtype=np.float64)
