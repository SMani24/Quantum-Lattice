# Phase 0: reproduce Figures 11 and 12

Status: planned; runner and figures are not implemented yet.

Use [configs/phase0_figures_11_12.json](../../configs/phase0_figures_11_12.json). It declares our initial settings rather than claiming recovery of the original missing grid/sample counts.

Reproduce the numerical method and trends in Appendix D.1:

1. Draw independent Bernoulli Z-error flags on periodic edge lattices.
2. Count odd-parity star syndromes and estimate their mean fraction p, with uncertainty across complete independent lattice draws.
3. Produce Figure 11 curves for L=8,12,16,20,24,28,32.
4. Fit a fresh zero-intercept fourth-degree inverse e_r(p) to L=32 data and produce Figure 12.
5. Compare samples and the new fit with published D4 and our derived exact independent-error relation.

The axes are horizontal p and vertical e_r. The paper's Figure 11 caption reverses them; our captions should use the plotted labels.

## Deliverables

- Separate Figure 11/12 PDF, SVG, and 300-DPI PNG exports.
- A two-page presentation PDF with explanatory captions.
- Numeric data, uncertainties, new fit coefficients, residuals, and fitting MSE.
- A short explanation of probabilities, assumptions, source, and reproduction differences.
- Effective config, random-stream/seed records, software versions, and a rerun command.

Store these under codex material/artifacts/phase0_figures_11_12/<run-id>/ with data/, figures/, presentation/, and metadata.

## Completion checks

- Zero errors produce no star defects.
- Every toric error pattern has an even total star-defect count.
- Sample means agree with p=[1-(1-2e_r)^4]/2 within quantified uncertainty.
- Lattice-size differences are assessed against that uncertainty.
- Fresh fit, published polynomial, and analytic check are clearly labeled.
- Fitting errors and low-rate behavior are reported; identical published coefficients/MSE are not required.
- Exports are visually inspected for legibility, clipping, and correct labels.

This is independent-Pauli-error calibration, separate from Metropolis ground-state sampling.
