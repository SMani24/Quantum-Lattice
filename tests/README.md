# Checks

Use unit/ for geometry, parity, basis/observable identities, estimators, and metrics. Use integration/ for small seeded experiment/dataset pipelines.

Phase 0 checks should cover zero-error behavior, even total torus syndrome parity, and agreement with the independent-error analytic relation.

Run the unit and integration checks from the repository root:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The calibration tests cover periodic edge incidence, syndrome cancellation, exact probability/variance identities, reproducible sampling, and polynomial fitting. Earlier author-code audits remain under codex material/code-reference/validation/.
