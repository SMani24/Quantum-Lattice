"""Geometry constructors for the shared immutable lattice representation."""

from __future__ import annotations

from .square import validate_side_length
from .topology import Lattice


def square_lattice(side_length: int) -> Lattice:
    """Build a periodic square torus matching the legacy array convention.

    Vertex ID is ``y * L + x``. Edge ID is ``2 * (y * L + x) + orientation``:
    0 is horizontal toward x+1, and 1 is vertical toward y+1. Positions are
    (row, column). Face ID y*L+x has the four ordered perimeter edges around
    the cell anchored at (y, x), with all coordinates wrapping modulo L.
    """
    length = validate_side_length(side_length)

    def vertex(y: int, x: int) -> int:
        return (y % length) * length + (x % length)

    def edge(y: int, x: int, orientation: int) -> int:
        return 2 * vertex(y, x) + orientation

    endpoints = []
    faces = []
    positions = []
    for y in range(length):
        for x in range(length):
            positions.append((y, x))
            endpoints.append((vertex(y, x), vertex(y, x + 1)))
            endpoints.append((vertex(y, x), vertex(y + 1, x)))
            faces.append(
                (edge(y, x, 0), edge(y, x + 1, 1), edge(y + 1, x, 0), edge(y, x, 1))
            )
    return Lattice(
        length * length,
        endpoints,
        faces,
        positions=positions,
        geometry="square_toric",
        boundary="periodic",
        shape=(length, length),
    )
