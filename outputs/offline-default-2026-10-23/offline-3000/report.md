# Matched reference: 3,000 simulations

nested fixed training banks; shared independent validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.336428  0.397753  0.235388  0.410399
Log-gamma              -inf      -inf      -inf      -inf
ECE                0.436667  0.450000  0.430000  0.438333
Post. Contraction  0.995354  0.995726  0.996008  0.993021

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
