# Matched reference: 480,000 simulations

fresh online simulations; shared validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.327966  0.747364  0.253347  0.434908
Log-gamma          4.031372  3.600774  2.330891  3.911272
ECE                0.014298  0.033158  0.034825  0.008158
Post. Contraction  0.904142  0.473591  0.937580  0.800413

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
