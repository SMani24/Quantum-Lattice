# Experiments

Each experiment folder owns a thin runner and description. Shared calculations belong in src/quantum_lattice; settings belong in configs; outputs belong in codex material/artifacts.

| Folder | Planned work |
|---|---|
| phase0_figures_11_12/ | Numerical calibration and professor-facing plots |
| baseline_mlp/ | Validated data generation, local MLP, and correction baseline |
| observable_cnn/ | Equal-information CNN, followed by signed-input comparisons |
| inverse_ising/ | Physics-based signed inference in the solvable branch |
| snapshot_models/ | Finite-shot observables versus raw snapshot sets |
| transfer_and_feedback/ | Unseen sizes/distributions and repeated compensation |

Only descriptions and the Phase 0 config exist now. Runners will be implemented as phases begin.

Every run should have a unique output directory containing effective config, environment/provenance metadata, numeric results, and plots. Keep earlier runs when changing settings.
