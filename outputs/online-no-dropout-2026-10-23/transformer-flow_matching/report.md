# Matched reference: 480,000 simulations

fresh online simulations; shared validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.306167  0.726431  0.246406  0.416190
Log-gamma          0.751062  2.337544 -0.039352  2.642440
ECE                0.015175  0.040175  0.010877  0.010877
Post. Contraction  0.908367  0.496134  0.938553  0.817426

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
