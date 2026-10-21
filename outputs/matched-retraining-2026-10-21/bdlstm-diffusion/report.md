# Matched reference: 480,000 simulations

fresh online simulations; shared validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v      noise
NRMSE              0.698388  0.894671  1.336993   0.872859
Log-gamma              -inf -1.003668      -inf -32.518977
ECE                0.315000  0.034825  0.411667   0.164561
Post. Contraction  0.870940  0.297611  0.851019   0.503658

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
