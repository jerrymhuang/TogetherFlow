# Revised main-paper offline results

Sources: `outputs/main-paper-revision`. Both runs are complete; the original paper fragment remains unchanged.

The 3K bank is nested in the 30K bank. Validation and test banks are shared and use distinct seed ranges. Manifests, epoch histories and metrics are included here; raw banks, posterior samples and checkpoints remain in the source folders. All figures and primary metrics use the same posterior samples. Primary NRMSE uses the empirical-prior bootstrap normalization with a shared bootstrap seed; range-normalized scores are supplied separately and retain the current median aggregation.

The fragment updates the simulation study. The main model and inference-method descriptions must also describe the corrected steering, diffusive noise, reference-distance observations, transformer network and revised metrics. A reference to the extension appendix remains unresolved in the standalone preview.

The 30K held-out sampling pass was restarted from the saved checkpoint after it stopped advancing. The recovery retained the sampler, batch size and shared random seed sequence and saved completed batches; the recovery metadata is included. No training was repeated.
