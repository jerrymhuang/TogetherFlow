# Matched reference: 3,000 simulations

nested fixed training banks; shared independent validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.453626  0.660684  0.363279  0.520356
Log-gamma              -inf      -inf      -inf      -inf
ECE                0.147895  0.122281  0.169386  0.174035
Post. Contraction  0.869626  0.705626  0.901891  0.803177

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
