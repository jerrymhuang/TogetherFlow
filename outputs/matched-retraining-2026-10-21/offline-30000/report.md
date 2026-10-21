# Matched reference: 30,000 simulations

nested fixed training banks; shared independent validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.177264  0.309334  0.141103  0.257297
Log-gamma              -inf      -inf      -inf      -inf
ECE                0.333333  0.376667  0.256140  0.343333
Post. Contraction  0.989884  0.984960  0.990717  0.982742

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
