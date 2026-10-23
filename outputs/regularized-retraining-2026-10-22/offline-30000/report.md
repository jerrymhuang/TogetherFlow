# Matched reference: 30,000 simulations

nested fixed training banks; shared independent validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w          r         v      noise
NRMSE              0.226987   0.488711  0.181817   0.312406
Log-gamma         -5.573609 -11.982010 -5.024632 -12.473277
ECE                0.079649   0.077895  0.073509   0.127105
Post. Contraction  0.958279   0.840183  0.969724   0.916774

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
