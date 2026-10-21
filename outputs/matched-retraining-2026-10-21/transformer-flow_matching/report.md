# Matched reference: 480,000 simulations

fresh online simulations; shared validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.332113  0.736400  0.242468  0.431053
Log-gamma          1.742610  2.267438  1.332978  3.938610
ECE                0.011228  0.039211  0.031930  0.008860
Post. Contraction  0.899686  0.459668  0.938983  0.801646

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
