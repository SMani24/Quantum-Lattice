"""Diagnostic parallel tempering within one solvable reference star orbit.

Each temperature has its own local Metropolis RNG. Adjacent exchanges move
only configurations and walker labels, leaving RNGs and counters attached to
temperatures. This module does not replace the production bank sampler.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from quantum_lattice.lattice import Lattice
from quantum_lattice.simulation.solvable import (
    StarMetropolisChain, _fields, _nonnegative_integer,
)


_SEED_DOMAIN = 2026101617


def _seed(value: Any) -> int | list[int]:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("seed must be a nonempty list or nonnegative integer")
        return [_nonnegative_integer(item, "seed entry") for item in value]
    return _nonnegative_integer(value, "seed")


def _ladder(values: Any) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError("betas must be real and strictly increasing from 0 to 1")
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or len(raw) < 2:
        raise ValueError("betas must be real and strictly increasing from 0 to 1")
    betas = np.array(raw, dtype=float, copy=True)
    if (not np.isfinite(betas).all() or betas[0] != 0 or betas[-1] != 1
            or np.any(np.diff(betas) <= 0)):
        raise ValueError("betas must be real and strictly increasing from 0 to 1")
    betas.setflags(write=False)
    return betas


def _swap_log_ratio(beta_i: float, beta_j: float, score_i: float, score_j: float) -> float:
    """Log product-target ratio for exchanging two configurations."""
    return (beta_i - beta_j) * (score_j - score_i)


class ReplicaExchangeChain:
    """Fixed-ladder replica exchange with exact JSON checkpoint continuation.

    ``P_beta(z)`` is proportional to ``exp(beta * sum_e b_e*z_e)``.
    One local sweep updates every temperature chain by V random proposals.
    After every ``swap_interval_sweeps`` global sweeps, alternate even/odd
    disjoint adjacent pairs attempt exchanges. No adaptive ladder is used.

    Visits count walker locations initially and after each exchange round.
    A round trip starts on a walker's first visit to beta=1, subsequently
    reaches beta=0, and completes when that walker returns to beta=1.
    Round trips and acceptance are diagnostics, not mixing certificates.
    """

    def __init__(self, lattice: Lattice, fields: Any, seed: Any, *,
                 betas: Any = (0., .2, .4, .6, .8, 1.),
                 backend: str = "auto", lazy_probability: float = .05,
                 swap_interval_sweeps: int = 10, start: str = "random"):
        self.lattice = lattice
        self.fields = _fields(lattice, fields)
        self.seed = _seed(seed)
        self.betas = _ladder(betas)
        self.swap_interval_sweeps = _nonnegative_integer(swap_interval_sweeps, "swap interval")
        if not self.swap_interval_sweeps:
            raise ValueError("swap interval must be positive")
        if not isinstance(start, str) or start not in {"plus", "random"}:
            raise ValueError("start must be plus or random")
        if (isinstance(lazy_probability, (bool,np.bool_))
                or not isinstance(lazy_probability, (int,float,np.integer,np.floating))
                or not np.isfinite(lazy_probability) or not 0 <= lazy_probability < 1):
            raise ValueError("lazy_probability must be finite and in [0,1)")
        self.start = start
        self.chains = []
        for i, beta in enumerate(self.betas):
            proposal_seed = np.random.SeedSequence(self.seed, spawn_key=(_SEED_DOMAIN, 1, i))
            chain = StarMetropolisChain(lattice, beta*self.fields, proposal_seed,
                                        backend=backend, lazy_probability=lazy_probability)
            if start == "random":
                initial_rng = np.random.default_rng(np.random.SeedSequence(
                    self.seed, spawn_key=(_SEED_DOMAIN, 2, i)))
                theta = initial_rng.choice(np.array([-1, 1], dtype=np.int8), size=lattice.n_vertices)
                ends = lattice.edge_vertex_indices
                chain.spins[:] = theta[ends[:, 0]] * theta[ends[:, 1]]
            self.chains.append(chain)
        self.lazy_probability = float(lazy_probability)
        self.swap_rng = np.random.default_rng(np.random.SeedSequence(
            self.seed, spawn_key=(_SEED_DOMAIN, 0)))
        count = len(self.chains)
        self.sweeps = self.exchange_rounds = 0
        self.walker_labels = np.arange(count, dtype=np.int64)
        self.swap_attempts = np.zeros(count-1, dtype=np.int64)
        self.swap_accepts = np.zeros(count-1, dtype=np.int64)
        self.visits = np.zeros((count, count), dtype=np.int64)
        self.roundtrips = np.zeros(count, dtype=np.int64)
        self._roundtrip_phase = np.zeros(count, dtype=np.int8)
        self._record_visits()

    @property
    def target_spins(self) -> np.ndarray:
        """Current beta=1 configuration; callers should copy retained records."""
        return self.chains[-1].spins

    @property
    def spins(self) -> np.ndarray:
        return self.target_spins

    def _record_visits(self) -> None:
        self.visits[self.walker_labels, np.arange(len(self.chains))] += 1
        hot, cold = int(self.walker_labels[0]), int(self.walker_labels[-1])
        if self._roundtrip_phase[hot] == 1:
            self._roundtrip_phase[hot] = 2
        if self._roundtrip_phase[cold] == 2:
            self.roundtrips[cold] += 1
        self._roundtrip_phase[cold] = 1

    def _exchange(self) -> None:
        scores = np.asarray([float(self.fields @ chain.spins) for chain in self.chains])
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite replica-exchange score")
        for i in range(self.exchange_rounds % 2, len(self.chains)-1, 2):
            j = i+1
            log_ratio = _swap_log_ratio(self.betas[i], self.betas[j], scores[i], scores[j])
            uniform = self.swap_rng.random()
            self.swap_attempts[i] += 1
            if log_ratio >= 0 or uniform == 0 or math.log(uniform) < log_ratio:
                previous = self.chains[i].spins.copy()
                self.chains[i].spins[:] = self.chains[j].spins
                self.chains[j].spins[:] = previous
                self.walker_labels[i], self.walker_labels[j] = self.walker_labels[j], self.walker_labels[i]
                self.swap_accepts[i] += 1
        self.exchange_rounds += 1
        self._record_visits()

    def advance(self, sweeps: int) -> None:
        """Advance local sweeps and fixed exchange boundaries, independent of chunks."""
        remaining = _nonnegative_integer(sweeps, "sweeps")
        while remaining:
            until_swap = self.swap_interval_sweeps - self.sweeps % self.swap_interval_sweeps
            count = min(remaining, until_swap)
            for chain in self.chains:
                chain.advance(count)
            self.sweeps += count
            remaining -= count
            if self.sweeps % self.swap_interval_sweeps == 0:
                self._exchange()

    def diagnostics(self) -> dict:
        """JSON-safe per-temperature and per-walker descriptive diagnostics."""
        attempts = self.swap_attempts.tolist()
        accepts = self.swap_accepts.tolist()
        return {
            "sweeps": self.sweeps, "betas": self.betas.tolist(),
            "exchange_rounds": self.exchange_rounds,
            "swap_attempts": attempts, "swap_accepts": accepts,
            "swap_acceptance": [b/a if a else None for a,b in zip(attempts, accepts)],
            "walker_labels": self.walker_labels.tolist(),
            "visits": self.visits.tolist(), "roundtrips": self.roundtrips.tolist(),
            "hot_visits": self.visits[:, 0].tolist(),
            "cold_visits": self.visits[:, -1].tolist(),
            "local_attempted": [c.attempted for c in self.chains],
            "local_accepted": [c.accepted for c in self.chains],
            "local_lazy": [c.lazy for c in self.chains],
            "limitations": "Fixed-ladder diagnostic; visits and round trips do not certify mixing.",
        }

    def snapshot(self) -> dict:
        """Independent, JSON-compatible snapshot including every RNG state."""
        return {
            "version": 1, "fields": self.fields.tolist(), "seed": copy.deepcopy(self.seed),
            "betas": self.betas.tolist(), "lazy_probability": self.lazy_probability,
            "swap_interval_sweeps": self.swap_interval_sweeps, "start": self.start,
            "sweeps": self.sweeps, "exchange_rounds": self.exchange_rounds,
            "chains": [chain.snapshot() for chain in self.chains],
            "swap_rng_state": copy.deepcopy(self.swap_rng.bit_generator.state),
            "walker_labels": self.walker_labels.tolist(),
            "swap_attempts": self.swap_attempts.tolist(), "swap_accepts": self.swap_accepts.tolist(),
            "visits": self.visits.tolist(), "roundtrips": self.roundtrips.tolist(),
            "roundtrip_phase": self._roundtrip_phase.tolist(),
        }

    @classmethod
    def from_snapshot(cls, lattice: Lattice, fields: Any, snapshot: dict, *,
                      backend: str = "auto") -> ReplicaExchangeChain:
        if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
            raise ValueError("unsupported replica-exchange checkpoint")
        chain = cls(lattice, fields, snapshot.get("seed"), betas=snapshot.get("betas"),
                    backend=backend, lazy_probability=snapshot.get("lazy_probability"),
                    swap_interval_sweeps=snapshot.get("swap_interval_sweeps"),
                    start=snapshot.get("start"))
        if not np.array_equal(snapshot.get("fields"), chain.fields):
            raise ValueError("replica-exchange checkpoint fields differ")
        count = len(chain.chains)
        saved_chains = snapshot.get("chains")
        if not isinstance(saved_chains, list) or len(saved_chains) != count:
            raise ValueError("replica-exchange checkpoint chain count differs")
        chain.chains = [StarMetropolisChain.from_snapshot(
            lattice, beta*chain.fields, saved, backend=backend,
            lazy_probability=chain.lazy_probability)
            for beta,saved in zip(chain.betas, saved_chains)]
        chain.sweeps = _nonnegative_integer(snapshot.get("sweeps"), "sweeps")
        chain.exchange_rounds = _nonnegative_integer(snapshot.get("exchange_rounds"), "exchange rounds")
        if (chain.exchange_rounds != chain.sweeps // chain.swap_interval_sweeps
                or any(c.attempted != chain.sweeps*lattice.n_vertices for c in chain.chains)):
            raise ValueError("replica-exchange checkpoint sweep counters disagree")
        def array(name, shape):
            raw = np.asarray(snapshot.get(name))
            if raw.shape != shape or raw.dtype.kind not in "iu" or np.any(raw < 0):
                raise ValueError(f"invalid replica-exchange checkpoint {name}")
            return raw.astype(np.int64)
        chain.walker_labels = array("walker_labels", (count,))
        if not np.array_equal(np.sort(chain.walker_labels), np.arange(count)):
            raise ValueError("checkpoint walker labels must be a permutation")
        chain.swap_attempts = array("swap_attempts", (count-1,))
        chain.swap_accepts = array("swap_accepts", (count-1,))
        expected = np.array([(chain.exchange_rounds+1-i%2)//2 for i in range(count-1)])
        if (not np.array_equal(chain.swap_attempts, expected)
                or np.any(chain.swap_accepts > chain.swap_attempts)):
            raise ValueError("checkpoint swap counters disagree")
        chain.visits = array("visits", (count,count))
        if (np.any(chain.visits.sum(axis=0) != chain.exchange_rounds+1)
                or np.any(chain.visits.sum(axis=1) != chain.exchange_rounds+1)
                or np.any(chain.visits[chain.walker_labels,np.arange(count)] < 1)):
            raise ValueError("checkpoint visitation counters disagree")
        chain.roundtrips = array("roundtrips", (count,))
        phase = array("roundtrip_phase", (count,))
        if (np.any(phase > 2) or np.any(chain.roundtrips > chain.visits[:,0])
                or np.any(chain.roundtrips > np.maximum(0,chain.visits[:,-1]-1))):
            raise ValueError("checkpoint round-trip counters disagree")
        chain._roundtrip_phase = phase.astype(np.int8)
        rng_state = snapshot.get("swap_rng_state")
        if not isinstance(rng_state,dict) or rng_state.get("bit_generator") != "PCG64":
            raise ValueError("invalid replica-exchange swap RNG state")
        chain.swap_rng.bit_generator.state = copy.deepcopy(rng_state)
        return chain
