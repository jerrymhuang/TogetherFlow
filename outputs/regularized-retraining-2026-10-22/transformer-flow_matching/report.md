# Matched reference: 480,000 simulations

fresh online simulations; shared validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.622659  0.805472  0.457241  0.667380
Log-gamma          1.158767  3.465252  3.842709  4.067481
ECE                0.011842  0.021228  0.022281  0.015000
Post. Contraction  0.632426  0.404584  0.787252  0.510317

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
