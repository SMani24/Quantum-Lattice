# Experiments

Each experiment folder owns a thin runner and description. Shared calculations belong in src/quantum_lattice; settings belong in configs; outputs belong in codex material/artifacts.

| Folder | Planned work |
|---|---|
| phase0_figures_11_12/ | Numerical calibration and professor-facing plots |
| phase1_sample_generation/ | Resumable sampling and selectable paper-domain datasets |
| baseline_mlp/ | Later local MLP and correction baseline |
| estimator_validation/ | Paired-estimator census, exact checks and resumable independent-chain tests |
| observable_cnn/ | Equal-information CNN, followed by signed-input comparisons |
| inverse_ising/ | Physics-based signed inference in the solvable branch |
| snapshot_models/ | Finite-shot observables versus raw snapshot sets |
| transfer_and_feedback/ | Unseen sizes/distributions and repeated compensation |

Phase 0 and Phase 1 sample-generation runners are implemented. The Phase 1
dashboard selects one size at a time and displays completed/paused datasets;
the MLP queue and separate estimator-validation runner are also implemented.

Every run should have a unique output directory containing effective config, environment/provenance metadata, numeric results, and plots. Keep earlier runs when changing settings.
