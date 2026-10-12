# Checks

Use unit/ for geometry, parity, basis/observable identities, estimators, and metrics. Use integration/ for small seeded experiment/dataset pipelines.

Phase 0 has 18 unit/integration checks covering zero-error behavior, even total torus syndrome parity, and agreement with the independent-error analytic relation.

Run the unit and integration checks from the repository root:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The calibration tests cover periodic edge incidence, syndrome cancellation, exact probability/variance identities, reproducible sampling, and polynomial fitting. Exhaustive enumeration of all 2^18 L=3 edge-error patterns checks the covariance-aware whole-pattern variance. Integration checks run a small seeded experiment, inspect PDF/PNG signatures and metadata, regenerate exports without changing numeric data, and reject overwrites or outputs outside the ignored material folder. Earlier author-code audits remain under codex material/code-reference/validation/.
