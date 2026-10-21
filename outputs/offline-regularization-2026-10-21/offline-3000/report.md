# Matched reference: 3,000 simulations

nested fixed training banks; shared independent validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.453264  0.660018  0.361526  0.519836
Log-gamma              -inf      -inf      -inf      -inf
ECE                0.148509  0.119561  0.169561  0.177368
Post. Contraction  0.869200  0.701699  0.901434  0.802267

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
