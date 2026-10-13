# Experiment configurations

Store reproducible settings here. These files describe our experiments, not undocumented settings from the original paper.

The Phase 0 configuration is a declared starting point. Its grid and number of independent error patterns are our choices because the original calibration data and generator are missing. Refine sample counts if uncertainty checks require it, and retain the effective configuration with each run.

Future learning configs should identify lattice sizes, field distribution, phase/domain assumptions, measurement protocol, seeds, dataset split, model settings, metrics, and artifact paths.

Actual outputs go under codex material/artifacts/, not beside configurations.

For Phase 1, `phase1_paper_domain_campaign.json` defines all seven paper training
sizes and at least 7,450 independent training Hamiltonians per size.
`phase1_sample_generation.json` is its full L=4 profile;
`phase1_sample_generation_pilot.json` preserves the weak-field pilot.
The campaign distinguishes paper-supported settings from our split, sampling
protocol and compact-storage choices. The historical phase-selection/training
driver is unavailable.
