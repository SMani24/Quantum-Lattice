"""Deterministic whole-Hamiltonian vertex-gauge masks for raw-Z training.

A mask has eta_uv=g_u*g_v, so multiplying every record of a bag by eta
preserves its cycle sector and corresponds to fields b'=eta*b. Magnitude labels
are unchanged. No fields or labels enter this module. This equilibrium symmetry
does not establish that finite, fixed-start MCMC trajectories are stationary.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


GAUGE_SCHEMA_VERSION = 1
_DOMAIN = "quantum_lattice.vertex_gauge_masks.v1"


def _integer(value, name: str, minimum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def gauge_masks_from_vertices(vertex_signs, L: int) -> np.ndarray:
    """Return float32 edge masks (...,2,L,L) from vertex signs (...,L,L).

    Vertex (y,x) is y*L+x. Channel 0 leaves it toward (y,x+1), channel 1
    toward (y+1,x), both periodically. A global reversal of all vertex signs
    leaves its mask unchanged; this helper need not fix the root sign.
    """
    length = _integer(L, "L", 3)
    vertices = np.asarray(vertex_signs)
    if (vertices.ndim < 2 or vertices.shape[-2:] != (length, length)
            or vertices.size == 0 or vertices.dtype.kind not in "iuf"
            or not np.isin(vertices, (-1, 1)).all()):
        raise ValueError("vertex signs must be real +/-1 with shape (...,L,L)")
    vertices = vertices.astype(np.float32, copy=False)
    horizontal = vertices * np.roll(vertices, -1, axis=-1)
    vertical = vertices * np.roll(vertices, -1, axis=-2)
    return np.stack((horizontal, vertical), axis=-3)


def vertex_gauge_masks(uids, L: int, *, seed: int, epoch: int) -> np.ndarray:
    """Return stable float32 masks [H,2,L,L], one for each UID/epoch key.

    ``epoch`` is a nonnegative, zero-based optimizer-epoch index. The stream
    for a Hamiltonian depends only on schema, L, seed, epoch and its exact UID;
    batch order/chunking does not affect masks. Global NumPy/Torch RNG states
    are untouched. Repeated UIDs receive identical masks.

    Fix g_0=+1 and draw every other vertex sign uniformly and independently
    with a private PCG64 generator. This samples the distinct vertex-generated
    edge masks uniformly. Apply its single mask to all records of that H bag,
    or directly to the complete-edge means. Do not mask records independently.
    """
    length = _integer(L, "L", 3)
    seed = _integer(seed, "seed", 0)
    epoch = _integer(epoch, "epoch", 0)
    if isinstance(uids, (str, bytes)):
        raise ValueError("uids must be a nonempty one-dimensional string sequence")
    identifiers = np.asarray(uids, dtype=object)
    if (identifiers.ndim != 1 or len(identifiers) == 0
            or any(not isinstance(uid, str) or not uid for uid in identifiers)):
        raise ValueError("uids must be a nonempty one-dimensional string sequence")
    vertices = np.ones((len(identifiers), length * length), dtype=np.int8)
    for index, uid in enumerate(identifiers):
        key = {"domain": _DOMAIN, "L": length, "seed": seed,
               "epoch": epoch, "uid": str(uid)}
        payload = json.dumps(key, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        stream_seed = int.from_bytes(hashlib.sha256(payload).digest(), "little")
        rng = np.random.Generator(np.random.PCG64(stream_seed))
        vertices[index, 1:] = rng.integers(0, 2, size=length * length - 1,
                                          dtype=np.int8) * 2 - 1
    return gauge_masks_from_vertices(vertices.reshape(-1, length, length), length)


def gauge_mask_spec() -> dict:
    """Owned JSON-safe recipe for checkpoint and source-provenance bindings."""
    return {
        "schema_version": GAUGE_SCHEMA_VERSION,
        "domain": _DOMAIN,
        "hash": "SHA256 of canonical JSON, sort_keys=True, separators=(',',':'), ensure_ascii=False, UTF-8",
        "hash_fields": ["domain", "L", "seed", "epoch", "uid"],
        "stream_seed": "little-endian unsigned integer from all SHA256 digest bytes",
        "generator": "NumPy Generator(PCG64(stream_seed))",
        "vertex_sampling": "root +1; remaining row-major signs 2*integers(0,2,dtype=int8)-1",
        "root_vertex": 0,
        "vertex_order": "y*L+x",
        "edge_channels": ["horizontal (y,x) to (y,x+1)",
                          "vertical (y,x) to (y+1,x)"],
        "boundary": "periodic",
        "output_layout": "H,2,L,L",
        "output_dtype": "float32",
        "scope": "one mask for the complete Hamiltonian bag",
        "epoch_index": "zero_based_optimizer_epoch",
        "uses_true_fields": False,
        "mutates_global_rng": False,
    }
