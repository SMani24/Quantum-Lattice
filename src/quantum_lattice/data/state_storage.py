"""Lossless edge-spin storage, independent of Monte Carlo or estimator logic.

Packed records store one bit per edge, with bit 1 representing spin -1. Edge
ordering is unchanged; unused high bits in the last byte must remain zero.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ENCODINGS = {"int8", "packed_bits"}


def encode_states(states: np.ndarray, encoding: str = "int8") -> dict[str, np.ndarray]:
    """Return NPZ-ready arrays; legacy int8 files retain their original layout."""
    if encoding not in ENCODINGS:
        raise ValueError("state_encoding must be int8 or packed_bits")
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] < 1 or states.dtype != np.int8 or not np.isin(states, [-1, 1]).all():
        raise ValueError("states must be an int8 (records, edges) array of spins +/-1")
    if encoding == "int8":
        return {"states": states}
    return {
        "states_packed": np.packbits(states == -1, axis=-1, bitorder="little"),
        "state_n_edges": np.array(states.shape[1], dtype=np.int64),
        "state_encoding": np.array("packed_bits"),
    }


def decode_states(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Validate and decode legacy or bit-packed records to int8 edge spins.

    Accepts an opened ``np.load`` archive or a dictionary of its arrays. Packed
    data include the exact edge count so padding never becomes physical spins.
    """
    if "states" in arrays:
        if any(key in arrays for key in ("states_packed", "state_n_edges", "state_encoding")):
            raise ValueError("ambiguous spin-state storage layout")
        states = np.asarray(arrays["states"])
        if states.ndim != 2 or states.shape[1] < 1 or states.dtype != np.int8 or not np.isin(states, [-1, 1]).all():
            raise ValueError("legacy states must be int8 spins +/-1")
        return states
    if not all(key in arrays for key in ("states_packed", "state_n_edges", "state_encoding")):
        raise ValueError("missing spin-state records or packed storage descriptors")
    encoding = np.asarray(arrays["state_encoding"])
    count = np.asarray(arrays["state_n_edges"])
    if encoding.shape != () or encoding.dtype.kind not in "US" or encoding.item() != "packed_bits":
        raise ValueError("invalid packed state encoding")
    if count.shape != () or count.dtype.kind not in "iu" or int(count) < 1:
        raise ValueError("invalid packed state edge count")
    n_edges = int(count)
    packed = np.asarray(arrays["states_packed"])
    if packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != (n_edges + 7) // 8:
        raise ValueError("packed state dimensions or dtype disagree with edge count")
    remainder = n_edges % 8
    if remainder and np.any(packed[:, -1] >> remainder):
        raise ValueError("nonzero padding bits in packed states")
    bits = np.unpackbits(packed, axis=-1, count=n_edges, bitorder="little")
    return (1 - 2 * bits.astype(np.int8)).astype(np.int8, copy=False)
