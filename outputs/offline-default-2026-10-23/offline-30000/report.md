# Matched reference: 30,000 simulations

nested fixed training banks; shared independent validation and test banks. Checkpoint selection: final epoch; no early stopping. Metrics and figures use the same saved posterior draws.

                          w         r         v     noise
NRMSE              0.154771  0.315638  0.147862  0.224224
Log-gamma              -inf      -inf      -inf      -inf
ECE                0.287807  0.396667  0.284474  0.321667
Post. Contraction  0.990133  0.985876  0.990970  0.984669

Primary NRMSE is prior-normalized; the separate range-normalized CSV is for auditing. These runs change the original simulator dynamics, observable inputs, and summary architecture, so comparison with the original paper cannot isolate the effect of training budget.
