# Checks

Use unit/ for geometry, parity, basis/observable identities, estimators, and metrics. Use integration/ for small seeded experiment/dataset pipelines.

The initial Phase 0 reproduction had 18 checks. The topology foundation adds object/connectivity, generic-moment, and sampling regression checks for 49 unit/integration tests in total.

Phase 1 adds solvable-state wavefunction/permutation oracles, detailed balance,
short-chain exact comparisons, checkpoint/RNG and backend parity, statistical
summaries, terminal restoration, and grouped-dataset pause/resume checks. These
use tiny fixtures and short validation chains, never the configured full pilot.

Run the unit and integration checks from the repository root:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The calibration tests cover periodic edge incidence, syndrome cancellation, exact probability/variance identities, reproducible sampling, and polynomial fitting. Exhaustive enumeration of all 2^18 L=3 edge-error patterns checks the covariance-aware whole-pattern variance. Integration checks run a small seeded experiment, inspect PDF/PNG signatures and metadata, regenerate exports without changing numeric data, and reject overwrites or outputs outside the ignored material folder. Earlier author-code audits remain under codex material/code-reference/validation/.

`test_topology.py` checks reciprocal object navigation, immutable incidence, periodic edge ordering, generic-versus-original square parity, star/plaquette commutation, and every error pattern on a tetrahedral non-square fixture. This fixture validates generic connectivity; it is not a honeycomb or triangular torus implementation.

`test_lattice_calibration.py` compares topology-derived means and variance with exhaustive weighted enumeration, including mixed vertex degrees, parallel edges, isolated vertices, and deterministic endpoints. `test_lattice_sampling.py` checks exact square random-stream/statistics equivalence and a generic non-square calibration point. The integration pipeline checks that one topology is constructed per size and reused across all probabilities.
