# Revised estimator architecture appendix

Use `estimator_architecture.tex` as the replacement section. Copy the files in
`figures/` into the manuscript figure directory, or add this directory to
`\graphicspath`. The section uses graphicx, booktabs, multirow, hyperref and
natbib, and retains the manuscript's existing wen2022transformers and
lipman2023flow citation keys. No new bibliography entries are required.

Sources: the four completed online architecture arms in
`outputs/matched-retraining-2026-10-21`. The offline regularization reruns and
Superstats experiments are excluded. The source manifest's generic `purpose`
field says offline recovery, but `variant.online=true`, training design, budgets,
and histories identify these four runs as online; the appendix uses those fields.

Important changes:
- Revised reference is transformer/flow matching; original recurrent architecture
  remains a comparison arm.
- Updated all 12 table rows, including ECE to expose overconfidence.
- Verified total network parameter counts by loading the saved models on CPU.
- Verified 95% coverage directly from saved physical-scale posterior samples.
- Replaced the old claims supporting recurrent summaries with the observed
  recovery/calibration failures, retaining the radius-NRMSE exception.
- Removed claims of universal flow-matching superiority and universal restrictions
  on compositional inference.
- Removed comparisons of absolute losses across different training objectives.
- Explicitly limited conclusions to one seed and the final-epoch online protocol.

`table_values.csv`, `model_and_coverage_audit.json`, per-arm manifests/metrics/
histories, and `figure_sources.csv` retain the numerical and figure provenance.
PDF and PNG figure assets are identical copies of the saved diagnostics.
A compiled appendix preview is not included because no LaTeX compiler is
available in this environment.
