# Phase 0: reproduce Figures 11 and 12

Implemented and validated with a full seeded run on 12 October 2026.

Use [configs/phase0_figures_11_12.json](../../configs/phase0_figures_11_12.json). It declares our settings rather than claiming recovery of the original missing grid/sample counts. After installing the package as described in the root README:

```sh
python experiments/phase0_figures_11_12/run.py --config configs/phase0_figures_11_12.json
```

Reproduce the numerical method and trends in Appendix D.1:

1. Draw independent Bernoulli Z-error flags on periodic edge lattices.
2. Count odd-parity star syndromes and estimate their mean fraction p, with uncertainty across complete independent lattice draws.
3. Produce Figure 11 curves for L=8,12,16,20,24,28,32.
4. Fit a fresh zero-intercept fourth-degree inverse e_r(p) to L=32 data and produce Figure 12.
5. Compare samples and the new fit with published D4 and our derived exact independent-error relation.

The axes are horizontal p and vertical e_r. The paper's Figure 11 caption reverses them; our captions should use the plotted labels.

## Sampling and fit

The default run uses 41 equally spaced error probabilities in [0, 0.2], 10,000 independent whole-lattice patterns per point, batches of 256, and seed 20261012. NumPy PCG64 streams use `SeedSequence(seed, spawn_key=(L, probability_grid_index))`. Edge arrays have shape `(patterns, L, L, 2)` for horizontal and vertical edges; star defects are the XOR of four incident error flags.

For independent errors, the exact forward relation is

```text
p = [1 - (1 - 2 e_r)^4] / 2
e_r = [1 - (1 - 2 p)^(1/4)] / 2, on 0 <= e_r <= 1/2.
```

Uncertainty is the sample standard deviation of independent whole-pattern defect fractions divided by the square root of the pattern count. Individual vertices within one pattern are correlated through shared edges. A separate exact variance check accounts for this covariance.

The fresh fit minimizes squared residuals in `e_r` using columns `p, p^2, p^3, p^4`, with no constant term. Published D4 coefficients are shown as a separate comparison. The fit is descriptive calibration; its in-sample MSE is not a held-out error or a neural-network accuracy measure. The polynomial is an approximate conversion, so the exact inverse is preferable for this strictly independent-error model.

## Deliverables

- Separate Figure 11/12 PDF, SVG, and 300-DPI PNG exports.
- A two-page presentation PDF with explanatory captions.
- Numeric data, uncertainties, new fit coefficients, residuals, and fitting MSE.
- A short explanation of probabilities, assumptions, source, and reproduction differences.
- Effective config, random-stream/seed records, software versions, and a rerun command.

Store these under codex material/artifacts/phase0_figures_11_12/<run-id>/ with data/, figures/, presentation/, and metadata.

Run IDs default to UTC timestamps. Sampling refuses an existing run directory; `INCOMPLETE` remains if a run fails. To regenerate exports from retained data without resampling:

```sh
python experiments/phase0_figures_11_12/run.py --render-only "codex material/artifacts/phase0_figures_11_12/<run-id>"
```

Both sampling and rendering require output beneath the local ignored material folder. Sampling and rendering retain separate source/input hashes.

## Initial validation

The retained seed-20261012 run contains 287 calibration points and 2,870,000 independent lattice patterns. The fresh L=32 inverse fit has MSE `1.72824e-7`; published D4 evaluated on our samples has MSE `1.73766e-7`. The largest forward-relation deviation is 3.352 standard errors among 280 stochastic points, with one beyond 3 SE and none beyond 5 SE. Individual exceedances are expected across many comparisons.

All 18 unit/integration checks pass. The separate figures and both presentation pages were visually inspected; the PDF has two landscape pages and embedded fonts. Generated artifacts and detailed run records remain local.

## Completion checks

- Zero errors produce no star defects.
- Every toric error pattern has an even total star-defect count.
- Sample means agree with p=[1-(1-2e_r)^4]/2 within quantified uncertainty.
- Lattice-size differences are assessed against that uncertainty.
- Fresh fit, published polynomial, and analytic check are clearly labeled.
- Fitting errors and low-rate behavior are reported; identical published coefficients/MSE are not required.
- Exports are visually inspected for legibility, clipping, and correct labels.

This is independent-Pauli-error calibration, separate from Metropolis ground-state sampling.
